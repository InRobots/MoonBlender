import torch

from pytorch3d.ops import sample_farthest_points

def seg_fps_gaussian(points, labels_one_hot, sample_percentages):

    # return mask
    B, N, C = points.shape
    num_classes = len(sample_percentages)
    assert num_classes == labels_one_hot.shape[1]
    device = points.device

    # 将 one-hot 标签转换为类别索引
    labels = torch.argmax(labels_one_hot, dim=1)  # [B, N]

    # 检查 sample_percentages 的长度
    if len(sample_percentages) != num_classes:
        raise ValueError("sample_percentages 的长度应等于类别数 num_classes")

    # 用于存储每个子点云的信息
    subclouds_points = []  # 存储每个子点云的点，列表长度为 num_subclouds
    subclouds_lengths = []  # 存储每个子点云的长度，列表长度为 num_subclouds
    subclouds_K = []        # 存储每个子点云的采样点数，列表长度为 num_subclouds
    subclouds_batch_idx = []  # 存储每个子点云对应的批次索引
    subclouds_original_indices = []  # 存储每个子点云中点在原始点云中的索引

    for b in range(B):
        for cls in range(num_classes):
            # 获取属于当前批次和类别的点的掩码
            mask = (labels[b] == cls)  # [N]
            cls_point_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)  # [N_cls]
            n_cls_points = cls_point_indices.shape[0]
            if n_cls_points == 0:
                continue  # 跳过没有该类别的情况

            # 计算采样的点数
            percentage = sample_percentages[cls]
            n_samples = max(int(n_cls_points * percentage + 0.5), 1)  # 至少采样1个点

            # 获取这些点的坐标
            cls_points = points[b, cls_point_indices, :]  # [N_cls, C]

            # 存储信息
            subclouds_points.append(cls_points)  # [N_cls, C]
            subclouds_lengths.append(n_cls_points)  # int
            subclouds_K.append(n_samples)  # int
            subclouds_batch_idx.append(b)  # int
            subclouds_original_indices.append(cls_point_indices)  # [N_cls]

    if len(subclouds_points) == 0:
        raise ValueError("No points to sample.")

    num_subclouds = len(subclouds_points)

    # 找到所有子点云中最大的点数
    P_max = max(subclouds_lengths)

    # 对每个子点云进行 padding，使其长度为 P_max
    padded_subclouds_points = []
    for pc in subclouds_points:
        n_points = pc.shape[0]
        pad_size = P_max - n_points
        if pad_size > 0:
            pad = torch.zeros((pad_size, C), device=device, dtype=pc.dtype)
            pc_padded = torch.cat([pc, pad], dim=0)  # [P_max, C]
        else:
            pc_padded = pc
        padded_subclouds_points.append(pc_padded)  # [P_max, C]

    # 将所有子点云组合成一个张量
    all_points = torch.stack(padded_subclouds_points, dim=0)  # [num_subclouds, P_max, C]
    all_lengths = torch.tensor(subclouds_lengths, dtype=torch.int64, device=device)  # [num_subclouds]
    all_K = torch.tensor(subclouds_K, dtype=torch.int64, device=device)  # [num_subclouds]

    # 调用 sample_farthest_points
    sampled_points, sampled_indices = sample_farthest_points(
        all_points,        # [num_subclouds, P_max, C]
        lengths=all_lengths,  # [num_subclouds]
        K=all_K,           # [num_subclouds]
    )  # 返回值形状为 [num_subclouds, max_K, C] 和 [num_subclouds, max_K]

    # 将采样的索引映射回原始点云的索引
    mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    for i in range(num_subclouds):
        b_idx = subclouds_batch_idx[i]
        original_indices = subclouds_original_indices[i]  # 原始点云中的索引 [N_cls]
        sampled_idx = sampled_indices[i]  # 采样得到的索引 [K_i]

        # 实际采样的点数
        n_samples = subclouds_K[i]

        # 过滤掉可能超出实际点数的索引
        valid_sampled_idx = sampled_idx[sampled_idx < subclouds_lengths[i]]

        # 映射回原始点云的索引
        selected_indices = original_indices[valid_sampled_idx]  # [n_valid_samples]

        # 更新掩码
        mask[b_idx, selected_indices] = True

    return mask  # [B, N]


def per_class_fps_percentage(points, labels_one_hot, sample_percentages):
    """
    在每个类别内按照指定的百分比进行最远点采样

    输入:
        points: 点云数据，[B, N, C]
        labels_one_hot: one-hot 编码的语义标签，[B, num_classes, N]
        sample_percentages: 列表，长度为 num_classes，每个元素为对应类别的采样百分比 (0~1)

    返回:
        mask: 采样结果的掩码，[B, N]
    """
    B, N, C = points.shape
    num_classes = labels_one_hot.shape[1]
    device = points.device

    # 将 one-hot 标签转换为类别索引
    labels = torch.argmax(labels_one_hot, dim=1)  # [B, N]

    # 检查 sample_percentages 的长度
    if len(sample_percentages) != num_classes:
        raise ValueError("sample_percentages 的长度应等于类别数 num_classes")

    # 用于存储每个子点云的信息
    subclouds_points = []  # 存储每个子点云的点，列表长度为 num_subclouds
    subclouds_lengths = []  # 存储每个子点云的长度，列表长度为 num_subclouds
    subclouds_K = []        # 存储每个子点云的采样点数，列表长度为 num_subclouds
    subclouds_batch_idx = []  # 存储每个子点云对应的批次索引
    subclouds_original_indices = []  # 存储每个子点云中点在原始点云中的索引

    for b in range(B):
        for cls in range(num_classes):
            # 获取属于当前批次和类别的点的掩码
            mask = (labels[b] == cls)  # [N]
            cls_point_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)  # [N_cls]
            n_cls_points = cls_point_indices.shape[0]
            if n_cls_points == 0:
                continue  # 跳过没有该类别的情况

            # 计算采样的点数
            percentage = sample_percentages[cls]
            n_samples = max(int(n_cls_points * percentage + 0.5), 1)  # 至少采样1个点

            # 获取这些点的坐标
            cls_points = points[b, cls_point_indices, :]  # [N_cls, C]

            # 存储信息
            subclouds_points.append(cls_points)  # [N_cls, C]
            subclouds_lengths.append(n_cls_points)  # int
            subclouds_K.append(n_samples)  # int
            subclouds_batch_idx.append(b)  # int
            subclouds_original_indices.append(cls_point_indices)  # [N_cls]

    if len(subclouds_points) == 0:
        raise ValueError("No points to sample.")

    num_subclouds = len(subclouds_points)

    # 找到所有子点云中最大的点数
    P_max = max(subclouds_lengths)

    # 对每个子点云进行 padding，使其长度为 P_max
    padded_subclouds_points = []
    for pc in subclouds_points:
        n_points = pc.shape[0]
        pad_size = P_max - n_points
        if pad_size > 0:
            pad = torch.zeros((pad_size, C), device=device, dtype=pc.dtype)
            pc_padded = torch.cat([pc, pad], dim=0)  # [P_max, C]
        else:
            pc_padded = pc
        padded_subclouds_points.append(pc_padded)  # [P_max, C]

    # 将所有子点云组合成一个张量
    all_points = torch.stack(padded_subclouds_points, dim=0)  # [num_subclouds, P_max, C]
    all_lengths = torch.tensor(subclouds_lengths, dtype=torch.int64, device=device)  # [num_subclouds]
    all_K = torch.tensor(subclouds_K, dtype=torch.int64, device=device)  # [num_subclouds]

    # 调用 sample_farthest_points
    sampled_points, sampled_indices = sample_farthest_points(
        all_points,        # [num_subclouds, P_max, C]
        lengths=all_lengths,  # [num_subclouds]
        K=all_K,           # [num_subclouds]
    )  # 返回值形状为 [num_subclouds, max_K, C] 和 [num_subclouds, max_K]

    # 将采样的索引映射回原始点云的索引
    mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    for i in range(num_subclouds):
        b_idx = subclouds_batch_idx[i]
        original_indices = subclouds_original_indices[i]  # 原始点云中的索引 [N_cls]
        sampled_idx = sampled_indices[i]  # 采样得到的索引 [K_i]

        # 实际采样的点数
        n_samples = subclouds_K[i]

        # 过滤掉可能超出实际点数的索引
        valid_sampled_idx = sampled_idx[sampled_idx < subclouds_lengths[i]]

        # 映射回原始点云的索引
        selected_indices = original_indices[valid_sampled_idx]  # [n_valid_samples]

        # 更新掩码
        mask[b_idx, selected_indices] = True

    return mask  # [B, N]


