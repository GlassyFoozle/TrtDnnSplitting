import math


class InferenceSegment:
    def __init__(self, G_segment, max_block_count, per_splitting_overhead=10):
        self.G_segment = G_segment
        self.max_block_count = max_block_count
        self.per_splitting_overhead = per_splitting_overhead
        self.base_block_list = []
        self.G_block_list = []
        self.splitting_config = []
        self.fixed_one_indices = set()
        self.no_overhead_split_indices = set()

        if G_segment < 1:
            print("[Error - InferenceSegment] G_segment < 1")
            return
        if max_block_count < 1:
            print("[Error - InferenceSegment] max_block_count < 1")
            return
        if G_segment < max_block_count:
            print("[Error - InferenceSegment] G_segment < max_block_count")
            print(f"G_segment: {G_segment}, max_block_count: {max_block_count}")
            return

        self.base_block_list = self._split_preserving_tail_small(G_segment, max_block_count)
        # Default: no split; all base blocks are merged into one GPU block.
        self.splitting_config = [0] * (max_block_count - 1)
        self.G_block_list = self._compute_block_list()

    def is_valid(self):
        return len(self.base_block_list) > 0 and len(self.G_block_list) > 0

    @property
    def size(self):
        """Current number of blocks."""
        return len(self.G_block_list)

    @property
    def overhead(self):
        return (self.size - 1) * self.per_splitting_overhead

    @property
    def G(self):
        """Total GPU time including splitting overhead."""
        return sum(self.G_block_list)

    @staticmethod
    def _split_preserving_tail_small(total, size):
        """Split total into `size` parts; last part is naturally smallest."""
        parts = []
        remain = total
        remain_slots = size
        for _ in range(size):
            part = math.ceil(remain / remain_slots)
            parts.append(part)
            remain -= part
            remain_slots -= 1
        return parts

    def _group_blocks(self, splitting_config):
        """Group base_block_list by splitting_config.

        splitting_config[i] = 1  ->  split between base_block[i] and base_block[i+1]
        splitting_config[i] = 0  ->  merge base_block[i+1] into current group
        """
        groups = []
        current_group = [self.base_block_list[0]]
        for i, split in enumerate(splitting_config):
            if split == 1:
                groups.append(current_group)
                current_group = [self.base_block_list[i + 1]]
            else:
                current_group.append(self.base_block_list[i + 1])
        groups.append(current_group)
        return groups

    def _compute_block_list(self):
        """Compute G_block_list from current splitting_config with overhead."""
        if not self.base_block_list:
            return []
        no_overhead_split_indices = getattr(self, "no_overhead_split_indices", set())
        result = []
        current_sum = self.base_block_list[0]
        for boundary_idx, split in enumerate(self.splitting_config):
            if split == 1:
                if boundary_idx not in no_overhead_split_indices:
                    current_sum += self.per_splitting_overhead
                result.append(current_sum)
                current_sum = self.base_block_list[boundary_idx + 1]
            else:
                current_sum += self.base_block_list[boundary_idx + 1]
        result.append(current_sum)
        return result

    def split_segment(self, n):
        """Split into n groups by distributing max_block_count base blocks evenly."""
        if not self.is_valid():
            return False
        if n < 1 or n > self.max_block_count:
            return False

        counts = self._split_preserving_tail_small(self.max_block_count, n)
        new_config = [0] * (self.max_block_count - 1)
        boundary = 0
        for count in counts[:-1]:
            boundary += count
            new_config[boundary - 1] = 1
        return self.split_by_config(new_config)

    def split_by_config(self, splitting_config):
        """Apply a splitting_config directly."""
        if not self.is_valid():
            return False
        expected_len = max(self.max_block_count - 1, 0)
        if len(splitting_config) != expected_len:
            return False
        if not all(c in (0, 1) for c in splitting_config):
            return False
        fixed_one_indices = getattr(self, "fixed_one_indices", set())
        if any(splitting_config[idx] != 1 for idx in fixed_one_indices):
            return False
        self.splitting_config = list(splitting_config)
        self.G_block_list = self._compute_block_list()
        return True

    def __repr__(self):
        return (
            f"InferenceSegment(G_segment={self.G_segment}, G={self.G}, "
            f"size={self.size}, max_block_count={self.max_block_count}, "
            f"overhead={self.overhead}, G_block_list={self.G_block_list})"
        )


class SegInfTask:
    def __init__(self, id, segment_list, period, deadline, priority, cpu=None):
        """
        segment_list format:
        [
            {
                'C': ...,
                'G_segment': ...,          # raw GPU time (no overhead); 0 for last CPU-only segment
                'max_block_count': ...,
                'per_splitting_overhead': ...,
            },
            ...
        ]
        """
        self.id = id
        self.cpu = cpu

        self.C_list = []
        self.G_segment_list = []       # list of G_block_lists per inference segment
        self.inference_segment_list = []

        for segment in segment_list:
            self.C_list.append(segment['C'])

            if segment['G_segment'] <= 0:  # Last (CPU-only) segment
                break

            inference_segment = InferenceSegment(
                segment['G_segment'],
                segment['max_block_count'],
                segment['per_splitting_overhead'],
            )

            if not inference_segment.is_valid():
                return

            self.inference_segment_list.append(inference_segment)
            self.G_segment_list.append(inference_segment.G_block_list)

        self.C = sum(self.C_list)
        self.G = sum(sum(blocks) for blocks in self.G_segment_list)
        self.max_block_count_list = [seg.max_block_count for seg in self.inference_segment_list]

        if self.G_segment_list:
            self.max_G_block = max(max(blocks) for blocks in self.G_segment_list)
        else:
            self.max_G_block = 0

        self.m = len(self.inference_segment_list) + 1  # number of execution segments

        self.T = self.period = period
        self.D = self.deadline = deadline
        self.pi = self.priority = priority

    def split_segment(self, idx, n):
        """Split inference segment at idx into n groups."""
        target = self.inference_segment_list[idx]
        if not target.split_segment(n):
            return False

        self.G_segment_list[idx] = target.G_block_list
        self.G = sum(sum(blocks) for blocks in self.G_segment_list)
        if self.G_segment_list:
            self.max_G_block = max(max(blocks) for blocks in self.G_segment_list)
        else:
            self.max_G_block = 0
        return True
    
    def convert_UNI_to_SS(self):
        if not getattr(self, "_UNI", False):
            message = "convert_UNI_to_SS() requires a UNI-converted task"
            print(f"[Error - SegInfTask] {message}")
            raise ValueError(message)

        block_sources = getattr(self, "_UNI_block_sources", None)
        if not block_sources:
            message = "convert_UNI_to_SS() requires UNI block source metadata"
            print(f"[Error - SegInfTask] {message}")
            raise ValueError(message)

        uni_segment = self.inference_segment_list[0]
        if not uni_segment.is_valid():
            return False

        original_segment_count = 0
        for source in block_sources:
            if source[0] == "G":
                original_segment_count = max(original_segment_count, source[1] + 1)

        c_list = [0] * (original_segment_count + 1)
        g_blocks_by_segment = [[] for _ in range(original_segment_count)]
        uni_pos_by_g_block = {}
        for pos, (value, source) in enumerate(zip(uni_segment.base_block_list, block_sources)):
            if source[0] == "C":
                c_list[source[1]] += value
            else:
                seg_idx = source[1]
                block_idx = source[2]
                g_blocks_by_segment[seg_idx].append(value)
                uni_pos_by_g_block[(seg_idx, block_idx)] = pos

        restored_segments = []
        for seg_idx, blocks in enumerate(g_blocks_by_segment):
            if not blocks:
                return False

            positive_total = max(sum(blocks), len(blocks))
            segment = InferenceSegment(
                positive_total,
                len(blocks),
                uni_segment.per_splitting_overhead,
            )
            segment.base_block_list = list(blocks)
            segment.max_block_count = len(blocks)
            segment.fixed_one_indices = set()
            segment.no_overhead_split_indices = set()
            segment.splitting_config = [0] * (len(blocks) - 1)
            for block_idx in range(len(blocks) - 1):
                left_pos = uni_pos_by_g_block[(seg_idx, block_idx)]
                right_pos = uni_pos_by_g_block[(seg_idx, block_idx + 1)]
                if right_pos == left_pos + 1:
                    segment.splitting_config[block_idx] = uni_segment.splitting_config[left_pos]
            segment.G_segment = sum(blocks)
            segment.G_block_list = segment._compute_block_list()
            restored_segments.append(segment)

        self.C_list = c_list
        self.inference_segment_list = restored_segments
        self.G_segment_list = [segment.G_block_list for segment in restored_segments]
        self.C = sum(c_list)
        self.G = sum(sum(blocks) for blocks in self.G_segment_list)
        self.max_block_count_list = [seg.max_block_count for seg in restored_segments]
        self.max_G_block = max(
            (max(blocks) for blocks in self.G_segment_list if blocks),
            default=0,
        )
        self.m = len(restored_segments) + 1
        self._UNI = False
        return True

    def convert_SS_to_UNI(self):
        if getattr(self, "_UNI", False):
            return True

        c_list = list(getattr(self, "C_list", []))
        segments = list(getattr(self, "inference_segment_list", []))
        base_blocks = []
        block_sources = []

        def append_block(value, source):
            if value is None or value <= 0:
                return
            base_blocks.append(value)
            block_sources.append(source)

        for seg_idx, segment in enumerate(segments):
            c_before = c_list[seg_idx] if seg_idx < len(c_list) else 0
            append_block(c_before, ("C", seg_idx))
            for block_idx, block in enumerate(segment.base_block_list):
                append_block(block, ("G", seg_idx, block_idx))

        final_c_idx = len(segments)
        final_c = c_list[final_c_idx] if final_c_idx < len(c_list) else 0
        append_block(final_c, ("C", final_c_idx))

        if not base_blocks:
            append_block(sum(c_list), ("C", 0))

        base_config = []
        fixed_one_indices = set()
        no_overhead_split_indices = set()
        for boundary_idx in range(max(len(base_blocks) - 1, 0)):
            left = block_sources[boundary_idx]
            right = block_sources[boundary_idx + 1]
            left_is_c = left[0] == "C"
            right_is_c = right[0] == "C"
            crosses_original_segment = (
                left[0] == "G"
                and right[0] == "G"
                and left[1] != right[1]
            )
            fixed = left_is_c or right_is_c or crosses_original_segment
            if fixed:
                base_config.append(1)
                fixed_one_indices.add(boundary_idx)
                no_overhead_split_indices.add(boundary_idx)
            elif left[0] == "G" and right[0] == "G" and left[1] == right[1]:
                base_config.append(segments[left[1]].splitting_config[left[2]])
            else:
                base_config.append(0)

        positive_total = max(sum(base_blocks), len(base_blocks))
        merged_segment = InferenceSegment(
            positive_total,
            len(base_blocks),
            segments[0].per_splitting_overhead if segments else 0,
        )
        merged_segment.base_block_list = list(base_blocks)
        merged_segment.max_block_count = len(base_blocks)
        merged_segment.fixed_one_indices = set(fixed_one_indices)
        merged_segment.no_overhead_split_indices = set(no_overhead_split_indices)
        merged_segment.splitting_config = list(base_config)
        merged_segment.G_segment = sum(base_blocks)
        merged_segment.G_block_list = merged_segment._compute_block_list()

        self.C_list = [0, 0]
        self.inference_segment_list = [merged_segment]
        self.G_segment_list = [merged_segment.G_block_list]
        self.C = 0
        self.G = sum(merged_segment.G_block_list)
        self.max_block_count_list = [merged_segment.max_block_count]
        self.max_G_block = max(merged_segment.G_block_list) if merged_segment.G_block_list else 0
        self.m = 2
        self._UNI = True
        self.non_splitting_config = list(base_config)
        self.fixed_one_indices = set(fixed_one_indices)
        self._UNI_block_sources = list(block_sources)
        return True

    def split_by_config(self, idx, splitting_config):
        """Apply splitting_config directly to inference segment at idx."""
        target = self.inference_segment_list[idx]
        if not target.split_by_config(splitting_config):
            return False

        self.G_segment_list[idx] = target.G_block_list
        self.G = sum(sum(blocks) for blocks in self.G_segment_list)
        if self.G_segment_list:
            self.max_G_block = max(max(blocks) for blocks in self.G_segment_list)
        else:
            self.max_G_block = 0
        return True

    def split_all_segments(self):
        """Split every inference segment to its maximum block count."""
        for idx, segment in enumerate(self.inference_segment_list):
            if segment.size == segment.max_block_count:
                continue
            if not self.split_segment(idx, segment.max_block_count):
                return False
        return True

    def is_valid(self):
        return len(self.inference_segment_list) > 0

    def __repr__(self):
        return (
            f"SegInfTask(id={self.id}, C={self.C}, G={self.G}, "
            f"max_G_block={self.max_G_block}, period={self.period}, "
            f"deadline={self.deadline}, priority={self.priority}, "
            f"segment_cnt={self.m}, cpu={self.cpu})"
        )
