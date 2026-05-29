import numpy as np


def compute_mrope_positions(
    *,
    input_ids: list[int],
    image_grid_thw: list[tuple[int, int, int]] | None,
    video_grid_thw: list[tuple[int, int, int]] | None,
    second_per_grid_ts: list[float] | None,
    vision_start_token_id: int,
    image_token_id: int,
    video_token_id: int | None,
    spatial_merge_size: int,
    tokens_per_second: int | float | None,
) -> tuple[np.ndarray, int]:
    input_tokens = list(input_ids)
    seq_len = len(input_tokens)
    if seq_len == 0:
        return np.zeros((3, 0), dtype=np.int32), 0

    tokens_per_second_val = 1.0 if tokens_per_second is None else float(tokens_per_second)
    image_grid_thw = image_grid_thw or []
    video_grid_thw = video_grid_thw or []
    second_per_grid_ts = second_per_grid_ts or []

    input_tokens_arr = np.asarray(input_tokens)
    vision_start_indices = np.argwhere(input_tokens_arr == vision_start_token_id).reshape(-1)
    image_nums = 0
    video_nums = 0
    if vision_start_indices.size > 0:
        vision_tokens = input_tokens_arr[vision_start_indices + 1]
        if vision_tokens.size > 0:
            image_nums = int(np.sum(vision_tokens == image_token_id))
            if video_token_id is not None:
                video_nums = int(np.sum(vision_tokens == video_token_id))

    llm_pos_ids_list: list[np.ndarray] = []
    st = 0
    remain_images, remain_videos = image_nums, video_nums
    image_index, video_index = 0, 0

    for _ in range(remain_images + remain_videos):
        ed_image = (
            input_tokens.index(image_token_id, st)
            if remain_images > 0 and image_token_id in input_tokens[st:]
            else len(input_tokens) + 1
        )
        ed_video = (
            input_tokens.index(video_token_id, st)
            if remain_videos > 0 and video_token_id in input_tokens[st:]
            else len(input_tokens) + 1
        )

        if ed_image < ed_video:
            t, h, w = image_grid_thw[image_index]
            second_per_grid_t = 0.0
            image_index += 1
            remain_images -= 1
            ed = ed_image
        else:
            t, h, w = video_grid_thw[video_index]
            second_per_grid_t = (
                float(second_per_grid_ts[video_index])
                if video_index < len(second_per_grid_ts)
                else 1.0
            )
            video_index += 1
            remain_videos -= 1
            ed = ed_video

        llm_grid_t = int(t)
        llm_grid_h = int(h // spatial_merge_size)
        llm_grid_w = int(w // spatial_merge_size)
        text_len = ed - st

        st_idx = int(llm_pos_ids_list[-1].max() + 1) if llm_pos_ids_list else 0
        if text_len > 0:
            text_positions = np.broadcast_to(
                np.arange(text_len, dtype=np.int32).reshape(1, -1), (3, text_len)
            )
            llm_pos_ids_list.append(text_positions + st_idx)

        t_index = (
            np.arange(llm_grid_t, dtype=np.float32).reshape(-1, 1)
            * second_per_grid_t
            * tokens_per_second_val
        ).astype(np.int32)
        t_index = np.broadcast_to(t_index, (llm_grid_t, llm_grid_h * llm_grid_w)).reshape(-1)
        h_index = np.broadcast_to(
            np.arange(llm_grid_h, dtype=np.int32).reshape(1, -1, 1),
            (llm_grid_t, llm_grid_h, llm_grid_w),
        ).reshape(-1)
        w_index = np.broadcast_to(
            np.arange(llm_grid_w, dtype=np.int32).reshape(1, 1, -1),
            (llm_grid_t, llm_grid_h, llm_grid_w),
        ).reshape(-1)
        vision_pos = np.stack([t_index, h_index, w_index], axis=0) + text_len + st_idx
        llm_pos_ids_list.append(vision_pos)

        st = ed + llm_grid_t * llm_grid_h * llm_grid_w

    if st < len(input_tokens):
        st_idx = int(llm_pos_ids_list[-1].max() + 1) if llm_pos_ids_list else 0
        text_len = len(input_tokens) - st
        text_positions = np.broadcast_to(
            np.arange(text_len, dtype=np.int32).reshape(1, -1), (3, text_len)
        )
        llm_pos_ids_list.append(text_positions + st_idx)

    llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
    mrope_position_delta = int(llm_positions.max() + 1 - len(input_tokens))
    return llm_positions.astype(np.int32), mrope_position_delta
