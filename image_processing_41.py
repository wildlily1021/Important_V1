import cv2
import numpy as np
from pathlib import Path
# from excelwrite import write_to_excel
global bandwidth, center_frequency

# def process_image(image_number):
def center_from_top_bottom_white(img,
                                 top_n=2,
                                 bottom_n=2,
                                 min_pixels_in_group=10,
                                 thresh_val=127,
                                 max_row_gap=5,              # 允许自动填补的最大行间断
                                 min_x_overlap_ratio=0.3,    # 横向重叠比例阈值
                                 debug=False,
                                 save_debug_path=None,       # 若不为 None，则保存 debug 可视化图（PNG/JPG 等）
                                 save_mask_path=None):       # 若不为 None，则保存用于计算的二值/填充掩码
    """
    在原有函数基础上新增了保存图像的能力：
      - save_debug_path: 若提供路径，则保存带质心/中心可视化的图像
      - save_mask_path: 若提供路径，则保存二值化或填充后的掩码图

    返回：
      debug=False: (cx, cy) 或 None
      debug=True:  (cx, cy, vis, selected_groups)
    （保存动作为副作用，不改变返回值结构）
    （这个算法也需要放进文件）
    """

    if img is None:
        return None

    arr = img.copy()
    if arr.ndim == 3 and arr.shape[2] == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    else:
        gray = arr if arr.ndim == 2 else cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    # 二值化
    _, bw = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    bw_bool = (bw == 255)
    H, W = bw_bool.shape

    # 找白色行
    rows_with_white = np.where(bw_bool.any(axis=1))[0]
    if rows_with_white.size == 0:
        if debug:
            return None, None, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), []
        return None

    splits = np.where(np.diff(rows_with_white) > 1)[0] + 1
    row_groups = np.split(rows_with_white, splits)

    # 每个 row group 的横向范围
    groups_info = []
    for g in row_groups:
        xs = []
        for y in g:
            cols = np.where(bw_bool[y, :])[0]
            if cols.size > 0:
                xs.extend(cols.tolist())
        if len(xs) < min_pixels_in_group:
            continue
        groups_info.append({
            'rows': g,
            'x_min': int(min(xs)),
            'x_max': int(max(xs))
        })

    if len(groups_info) == 0:
        # fallback global centroid
        ys_all, xs_all = np.where(bw_bool)
        if xs_all.size == 0:
            if debug:
                return None, None, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), []
            return None
        cx_global = float(xs_all.mean())
        cy_global = float(ys_all.mean())

        # save mask if requested
        if save_mask_path is not None:
            cv2.imwrite(save_mask_path, (bw.astype(np.uint8)))
        if debug:
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.circle(vis, (int(round(cx_global)), int(round(cy_global))), 5, (0,255,0), -1)
            if save_debug_path is not None:
                cv2.imwrite(save_debug_path, vis)
            return cx_global, cy_global, vis, [{'cx': cx_global, 'cy': cy_global, 'count': len(xs_all)}]
        return cx_global, cy_global

    # 自动合并“被黑线割裂”的白块
    merged_groups = [groups_info[0]]
    for cur in groups_info[1:]:
        prev = merged_groups[-1]
        row_gap = int(cur['rows'][0] - prev['rows'][-1] - 1)
        if row_gap <= max_row_gap:
            overlap = max(0, min(prev['x_max'], cur['x_max']) - max(prev['x_min'], cur['x_min']))
            # 宽度用较小的一个宽度作为归一化基准
            width = max(1.0, float(min(prev['x_max'] - prev['x_min'], cur['x_max'] - cur['x_min'])))
            overlap_ratio = overlap / width
            if overlap_ratio >= min_x_overlap_ratio:
                # 合并 cur 到 prev
                prev['rows'] = np.concatenate([prev['rows'], cur['rows']])
                prev['x_min'] = min(prev['x_min'], cur['x_min'])
                prev['x_max'] = max(prev['x_max'], cur['x_max'])
                continue
        merged_groups.append(cur)

    # 对合并后的 groups 计算质心
    group_infos = []
    # 创建一个填充后的掩码（可视化/保存用）：复制原始 bw_bool
    filled_mask = bw_bool.copy()
    for g in merged_groups:
        xs_all = []
        ys_all = []
        for y in g['rows']:
            cols = np.where(bw_bool[y, :])[0]
            if cols.size > 0:
                xs_all.append(cols)
                ys_all.append(np.full(cols.shape, y, dtype=int))
        if len(xs_all) == 0:
            continue
        xs_all = np.concatenate(xs_all)
        ys_all = np.concatenate(ys_all)
        if xs_all.size < min_pixels_in_group:
            continue
        group_infos.append({
            'cx': float(xs_all.mean()),
            'cy': float(ys_all.mean()),
            'count': int(xs_all.size),
            'rows': g['rows']
        })
        # 在 filled_mask 上把该 group 的行范围所有白像素设置成 True（用于视觉上的“填充”）
        for y in g['rows']:
            filled_mask[y, np.where(bw_bool[y, :])[0]] = True

    if len(group_infos) == 0:
        # fallback global centroid（与上同）
        ys_all, xs_all = np.where(bw_bool)
        if xs_all.size == 0:
            if debug:
                return None, None, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), []
            return None
        cx_global = float(xs_all.mean())
        cy_global = float(ys_all.mean())
        if save_mask_path is not None:
            cv2.imwrite(save_mask_path, (bw.astype(np.uint8)))
        if debug:
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.circle(vis, (int(round(cx_global)), int(round(cy_global))), 5, (0,255,0), -1)
            if save_debug_path is not None:
                cv2.imwrite(save_debug_path, vis)
            return cx_global, cy_global, vis, [{'cx': cx_global, 'cy': cy_global, 'count': len(xs_all)}]
        return cx_global, cy_global

    # 选 top + bottom
    group_infos_sorted = sorted(group_infos, key=lambda it: it['cy'])
    selected = group_infos_sorted[:top_n] + group_infos_sorted[-bottom_n:]

    # 去重
    unique = {}
    for g in selected:
        key = (round(g['cx'], 2), round(g['cy'], 2))
        if key not in unique:
            unique[key] = g
    selected = list(unique.values())

    cx_mean = float(np.mean([g['cx'] for g in selected]))
    cy_mean = float(np.mean([g['cy'] for g in selected]))

    # 保存掩码（如果请求）
    if save_mask_path is not None:
        # saved as 0/255 image
        to_save_mask = (filled_mask.astype(np.uint8) * 255)
        cv2.imwrite(save_mask_path, to_save_mask)

    if debug:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for g in selected:
            cv2.circle(vis, (int(round(g['cx'])), int(round(g['cy']))), 4, (0, 0, 255), -1)
        cv2.circle(vis, (int(round(cx_mean)), int(round(cy_mean))), 6, (0, 255, 0), -1)
        if save_debug_path is not None:
            cv2.imwrite(save_debug_path, vis)
        return cx_mean, cy_mean, vis, selected

    # 非 debug 模式仍然支持只保存掩码（或 debug 可视化）
    if save_debug_path is not None:
        # 生成一个简单的可视化图用于保存（即使 debug=False）
        vis_simple = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.circle(vis_simple, (int(round(cx_mean)), int(round(cy_mean))), 6, (0, 255, 0), -1)
        cv2.imwrite(save_debug_path, vis_simple)

    return cx_mean, cy_mean



def process_image(i, j, rec_wave, Fs, SNR, image_path_STFT):
#以前的视觉方法会忽略一些黑线，导致算法的中心频率估计出错，这是应对采样点数比较好的算法。
    global bandwidth, center_frequency

    def save_image(image, path):
        try:
            cv2.imwrite(path, image)
        except Exception as e:
            print(f"Error saving image: {e}")


    image_path = str(image_path_STFT)
    image_path_obj = Path(image_path)
    binary_output_path = str(image_path_obj.with_name(f"{image_path_obj.stem}_binary.jpg"))
    morph_output_path = str(image_path_obj.with_name(f"{image_path_obj.stem}_morph.jpg"))
    final_output_path = str(image_path_obj.with_name(f"{image_path_obj.stem}_final.jpg"))
    black_output_path = str(image_path_obj.with_name(f"{image_path_obj.stem}_black.jpg"))

    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image {image_path}")
        return

    # Get image dimensions
    height, width, channels = image.shape
    print(f"Image dimensions: {width}x{height} pixels")

    def photo_baoluo():
        # Create a binary mask for red colors based on RGB conditions
        red_mask_baoluo = np.logical_and(image[:, :, 1] < 82, image[:, :, 0] < 9)  # python是B,R,G,lmg!:.:2]代表R通道，也就是红色分量图像;lmg!::1]代表G通道，也就是绿色分量图像;Img[:,:,0]代表B通道，也就是蓝色分量图像
        red_mask_baoluo = red_mask_baoluo.astype(np.uint8) * 255

        # Save the binary image
        save_image(red_mask_baoluo, binary_output_path)

        # Apply morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        cleaned_image_baoluo = cv2.morphologyEx(red_mask_baoluo, cv2.MORPH_OPEN, kernel)
        cleaned_image_baoluo = cv2.morphologyEx(cleaned_image_baoluo, cv2.MORPH_CLOSE, kernel)

        # Save the image after morphological operations
        save_image(cleaned_image_baoluo, morph_output_path)

        # Convert each row to either fully black or fully white based on the percentage of black pixels
        for i in range(cleaned_image_baoluo.shape[0]):
            if np.mean(cleaned_image_baoluo[i, :]) < 85:  # More than 50% black pixels
                cleaned_image_baoluo[i, :] = 0  # Set entire row to black
            else:
                cleaned_image_baoluo[i, :] = 255  # Set entire row to white

        # Save the image after row-wise conversion
        save_image(cleaned_image_baoluo, final_output_path)

        # Find contours to calculate height and center point of the red region
        contours_baoluo, _ = cv2.findContours(cleaned_image_baoluo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_baoluo:
            mask = cv2.imread(final_output_path, cv2.IMREAD_GRAYSCALE)
            cx, cy, vis, groups = center_from_top_bottom_white(mask, debug=True)
            print("center:", cx, cy)
            cv2.imwrite('debug_center.png', vis)
            return [cx, cy]
        else:
            # Receiver data
            x_rec = rec_wave
            x_rec_con = np.conj(x_rec)
            L_rec = len(x_rec)

            # Autocorrelation calculations
            R_0 = np.sum(x_rec * x_rec_con) / L_rec  # R(0)
            R_1 = np.sum(x_rec[1:] * x_rec_con[:-1]) / (L_rec - 1)  # R(1)

            # Frequency search
            fre_num = 4096  # Number of frequency search points
            delta_f = Fs / 2 / fre_num  # Frequency search interval
            function_J_2 = np.zeros(int((Fs / 2 - delta_f) / delta_f) + 1, dtype=np.float64)  # Estimation function

            # Frequency search loop
            for idx, fre_find in enumerate(np.arange(delta_f, Fs / 2, delta_f)):
                function_J_2[idx] = np.real((R_1 * np.conj(R_0)) * np.exp(-1j * 2 * np.pi * fre_find / Fs)) ** 2

            # Maximum value corresponding to the center frequency
            J_max = np.max(function_J_2)
            idex_max = np.argmax(function_J_2)
            fre_center_esti = idex_max * delta_f
            center_frequency_output = (Fs - 2 * fre_center_esti) * height / Fs
            bandwidth_baoluo = 0
            return [bandwidth_baoluo, center_frequency_output / 1.001 * 0.993]

    # [bandwidth, center_frequency] = photo_signal()
    [bandwidth_b, center_frequency_b] = photo_baoluo()
    bandwidth_output = bandwidth_b
    center_frequency_output = center_frequency_b
    #处理中心频率偏差过高的问题：使用数据微调
    if j == 1:
        center_frequency_output = center_frequency_output * 0.98 / 1.02
    else:
        center_frequency_output = center_frequency_output
    # write_to_excel(row_num, height, bandwidth_output, center_frequency_output)
    return [height, center_frequency_output]

def process_image_test(rec_wave, Fs, image_path):
    global bandwidth, center_frequency

    def save_image(image, path):
        try:
            cv2.imwrite(path, image)
        except Exception as e:
            print(f"Error saving image: {e}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image {image_path}")
        return

    # Get image dimensions
    height, width, channels = image.shape
    print(f"Image dimensions: {width}x{height} pixels")

    def photo_signal():
        # Create a binary mask for red colors based on RGB conditions
        red_mask = np.logical_and(image[:, :, 1] < 10, image[:, :, 0] < 5) # python是B,G,R
        red_mask = red_mask.astype(np.uint8) * 255

        # Save the binary image
        # save_image(red_mask, binary_output_path)

        # Apply morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        cleaned_image = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        cleaned_image = cv2.morphologyEx(cleaned_image, cv2.MORPH_CLOSE, kernel)

        # Save the image after morphological operations
        # save_image(cleaned_image, morph_output_path)

        # Convert each row to either fully black or fully white based on the percentage of black pixels
        for i in range(cleaned_image.shape[0]):
            if np.mean(cleaned_image[i, :]) < 127:  # More than 50% black pixels
                cleaned_image[i, :] = 0  # Set entire row to black
            else:
                cleaned_image[i, :] = 255  # Set entire row to white

        # Save the image after row-wise conversion
        # save_image(cleaned_image, final_output_path)

        # Find contours to calculate height and center point of the red region
        contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            center_x = x + w / 2
            center_y = y + h / 2
            bandwidth = h
            center_frequency = center_y

            print(f"Bandwidth: {bandwidth:.2f}")
            print(f"Center Frequency: {center_frequency:.2f}")

            # Draw the bounding box and center point on the original image for visualization
            output_image = image.copy()
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(output_image, (int(center_x), int(center_y)), 5, (255, 0, 0), -1)
            # save_image(output_image, black_output_path)

            return [bandwidth, center_frequency]
            # write_to_excel(image_number, height, bandwidth, center_frequency)
        else:
            # 感觉小于100的时候就不用平均了
            return [0, 0]
            # print("No deep red region found in the image")
            # return

    def photo_baoluo():
        # Create a binary mask for red colors based on RGB conditions
        red_mask_baoluo = np.logical_and(image[:, :, 1] < 82, image[:, :, 0] < 9)  
        # python是B,R,G,
        # lmg!:.:2]代表R通道，也就是红色分量图像
        # lmg!::1]代表G通道，也就是绿色分量图像
        # Img[:,:,0]代表B通道，也就是蓝色分量图像
        red_mask_baoluo = red_mask_baoluo.astype(np.uint8) * 255

        # Save the binary image
        # save_image(red_mask_baoluo, binary_output_baoluo_path)

        # Apply morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        cleaned_image_baoluo = cv2.morphologyEx(red_mask_baoluo, cv2.MORPH_OPEN, kernel)
        cleaned_image_baoluo = cv2.morphologyEx(cleaned_image_baoluo, cv2.MORPH_CLOSE, kernel)

        # Save the image after morphological operations
        # save_image(cleaned_image_baoluo, morph_output_path)

        # Convert each row to either fully black or fully white based on the percentage of black pixels
        for i in range(cleaned_image_baoluo.shape[0]):
            if np.mean(cleaned_image_baoluo[i, :]) < 127:  # More than 50% black pixels
                cleaned_image_baoluo[i, :] = 0  # Set entire row to black
            else:
                cleaned_image_baoluo[i, :] = 255  # Set entire row to white

        # Save the image after row-wise conversion
        # save_image(cleaned_image_baoluo, final_output_path)

        # Find contours to calculate height and center point of the red region
        contours_baoluo, _ = cv2.findContours(cleaned_image_baoluo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_baoluo:
            largest_contour_baoluo = max(contours_baoluo, key=cv2.contourArea)
            x_baoluo, y_baoluo, w_baoluo, h_baoluo = cv2.boundingRect(largest_contour_baoluo)
            center_x_baoluo = x_baoluo + w_baoluo / 2
            center_y_baoluo = y_baoluo + h_baoluo / 2
            bandwidth_baoluo = h_baoluo
            center_frequency_baoluo = center_y_baoluo

            print(f"Bandwidth: {bandwidth_baoluo:.2f}")
            print(f"Center Frequency: {center_frequency_baoluo:.2f}")
            # Draw the bounding box and center point on the original image for visualization
            output_image = image.copy()
            cv2.rectangle(output_image, (x_baoluo, y_baoluo), (x_baoluo + w_baoluo, y_baoluo + h_baoluo), (0, 255, 0), 2)
            cv2.circle(output_image, (int(center_x_baoluo), int(center_y_baoluo)), 5, (255, 0, 0), -1)
            # save_image(output_image, black_output_path)
            return [bandwidth_baoluo, center_frequency_baoluo]
        else:
            # Receiver data
            x_rec = rec_wave
            x_rec_con = np.conj(x_rec)
            L_rec = len(x_rec)

            # Autocorrelation calculations
            R_0 = np.sum(x_rec * x_rec_con) / L_rec  # R(0)
            R_1 = np.sum(x_rec[1:] * x_rec_con[:-1]) / (L_rec - 1)  # R(1)

            # Frequency search
            fre_num = 4096  # Number of frequency search points
            delta_f = Fs / 2 / fre_num  # Frequency search interval
            function_J_2 = np.zeros(int((Fs / 2 - delta_f) / delta_f) + 1, dtype=np.float64)  # Estimation function

            # Frequency search loop
            for idx, fre_find in enumerate(np.arange(delta_f, Fs / 2, delta_f)):
                function_J_2[idx] = np.real((R_1 * np.conj(R_0)) * np.exp(-1j * 2 * np.pi * fre_find / Fs)) ** 2

            # Maximum value corresponding to the center frequency
            J_max = np.max(function_J_2)
            idex_max = np.argmax(function_J_2)
            fre_center_esti = idex_max * delta_f
            center_frequency_output = (Fs - 2 * fre_center_esti) * height / (2 * Fs)
            bandwidth_baoluo = 0
            return [bandwidth_baoluo, center_frequency_output / 1.001 * 0.993]

    # [bandwidth, center_frequency] = photo_signal()
    [bandwidth_b, center_frequency_b] = photo_baoluo()
    # bandwidth_output = bandwidth + bandwidth_b
    # center_frequency_output = center_frequency + center_frequency_b
    bandwidth_output = bandwidth_b * 2
    if center_frequency_b > height * 0.25:
        # Receiver data
        x_rec = rec_wave
        x_rec_con = np.conj(x_rec)
        L_rec = len(x_rec)

        # Autocorrelation calculations
        R_0 = np.sum(x_rec * x_rec_con) / L_rec  # R(0)
        R_1 = np.sum(x_rec[1:] * x_rec_con[:-1]) / (L_rec - 1)  # R(1)

        # Frequency search
        fre_num = 4096  # Number of frequency search points
        delta_f = Fs / 2 / fre_num  # Frequency search interval
        function_J_2 = np.zeros(int((Fs / 2 - delta_f) / delta_f) + 1, dtype=np.float64)  # Estimation function

        # Frequency search loop
        for idx, fre_find in enumerate(np.arange(delta_f, Fs / 2, delta_f)):
            function_J_2[idx] = np.real((R_1 * np.conj(R_0)) * np.exp(-1j * 2 * np.pi * fre_find / Fs)) ** 2

        # Maximum value corresponding to the center frequency
        J_max = np.max(function_J_2)
        idex_max = np.argmax(function_J_2)
        fre_center_esti = idex_max * delta_f
        center_frequency_output = (Fs - 2 * fre_center_esti) * height / (2 * Fs)
    else:
        center_frequency_output = center_frequency_b * 2 * 1.001 / 0.993

    # write_to_excel(row_num, height, bandwidth_output, center_frequency_output)
    return [height, center_frequency_output]

def process_image_final(Rs_process, image_path, rec_wave, Fs, SNR):
    global bandwidth, center_frequency

    def save_image(image, path):
        try:
            cv2.imwrite(path, image)
        except Exception as e:
            print(f"Error saving image: {e}")

    image_path = image_path

    # binary_output_path = f'./spectrogram_{image_number}_binary.jpg'
    # morph_output_path = f'./spectrogram_{image_number}_morph.jpg'
    # final_output_path = f'./spectrogram_{image_number}_final.jpg'
    # black_output_path = f'./spectrogram_{image_number}_black.jpg'

    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image {image_path}")
        return

    # Get image dimensions
    height, width, channels = image.shape
    print(f"Image dimensions: {width}x{height} pixels")

    def photo_baoluo():
        # Create a binary mask for red colors based on RGB conditions
        red_mask_baoluo = np.logical_and(image[:, :, 1] < 82, image[:, :, 0] < 9)  
        # python是B,R,G,lmg!:.:2]代表R通道，也就是红色分量图像;lmg!::1]代表G通道，也就是绿色分量图像;Img[:,:,0]代表B通道，也就是蓝色分量图像
        red_mask_baoluo = red_mask_baoluo.astype(np.uint8) * 255

        # Save the binary image
        # save_image(red_mask_baoluo, binary_output_baoluo_path)

        # Apply morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        cleaned_image_baoluo = cv2.morphologyEx(red_mask_baoluo, cv2.MORPH_OPEN, kernel)
        cleaned_image_baoluo = cv2.morphologyEx(cleaned_image_baoluo, cv2.MORPH_CLOSE, kernel)

        # Save the image after morphological operations
        # save_image(cleaned_image_baoluo, morph_output_path)

        # Convert each row to either fully black or fully white based on the percentage of black pixels
        for i in range(cleaned_image_baoluo.shape[0]):
            if np.mean(cleaned_image_baoluo[i, :]) < 127:  # More than 50% black pixels
                cleaned_image_baoluo[i, :] = 0  # Set entire row to black
            else:
                cleaned_image_baoluo[i, :] = 255  # Set entire row to white

        # Save the image after row-wise conversion
        # save_image(cleaned_image_baoluo, final_output_path)

        # Find contours to calculate height and center point of the red region
        contours_baoluo, _ = cv2.findContours(cleaned_image_baoluo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_baoluo:
            largest_contour_baoluo = max(contours_baoluo, key=cv2.contourArea)
            x_baoluo, y_baoluo, w_baoluo, h_baoluo = cv2.boundingRect(largest_contour_baoluo)
            center_x_baoluo = x_baoluo + w_baoluo / 2
            center_y_baoluo = y_baoluo + h_baoluo / 2
            bandwidth_baoluo = h_baoluo
            center_frequency_baoluo = center_y_baoluo

            print(f"Bandwidth: {bandwidth_baoluo:.2f}")
            print(f"Center Frequency: {center_frequency_baoluo:.2f}")
            # Draw the bounding box and center point on the original image for visualization
            output_image = image.copy()
            cv2.rectangle(output_image, (x_baoluo, y_baoluo), (x_baoluo + w_baoluo, y_baoluo + h_baoluo), (0, 255, 0), 2)
            cv2.circle(output_image, (int(center_x_baoluo), int(center_y_baoluo)), 5, (255, 0, 0), -1)
            # save_image(output_image, black_output_path)
            return [bandwidth_baoluo / 2.0, center_frequency_baoluo / 2.0]
        else:
            # Receiver data
            x_rec = rec_wave
            x_rec_con = np.conj(x_rec)
            L_rec = len(x_rec)

            # Autocorrelation calculations
            R_0 = np.sum(x_rec * x_rec_con) / L_rec  # R(0)
            R_1 = np.sum(x_rec[1:] * x_rec_con[:-1]) / (L_rec - 1)  # R(1)

            # Frequency search
            fre_num = 4096  # Number of frequency search points
            delta_f = Fs / 2 / fre_num  # Frequency search interval
            function_J_2 = np.zeros(int((Fs / 2 - delta_f) / delta_f) + 1, dtype=np.float64)  # Estimation function

            # Frequency search loop
            for idx, fre_find in enumerate(np.arange(delta_f, Fs / 2, delta_f)):
                function_J_2[idx] = np.real((R_1 * np.conj(R_0)) * np.exp(-1j * 2 * np.pi * fre_find / Fs)) ** 2

            # Maximum value corresponding to the center frequency
            J_max = np.max(function_J_2)
            idex_max = np.argmax(function_J_2)
            fre_center_esti = idex_max * delta_f
            center_frequency_output = (Fs - 2 * fre_center_esti) * height / (2 * Fs)
            bandwidth_baoluo = 0
            return [bandwidth_baoluo / 2.0, center_frequency_output / 2.0 / 1.001 * 0.993]

    # [bandwidth, center_frequency] = photo_signal()
    [bandwidth_b, center_frequency_b] = photo_baoluo()
    # bandwidth_output = bandwidth + bandwidth_b
    # center_frequency_output = center_frequency + center_frequency_b
    bandwidth_output = bandwidth_b * 2
    if center_frequency_b > height * 0.25:
        # Receiver data
        x_rec = rec_wave
        x_rec_con = np.conj(x_rec)
        L_rec = len(x_rec)

        # Autocorrelation calculations
        R_0 = np.sum(x_rec * x_rec_con) / L_rec  # R(0)
        R_1 = np.sum(x_rec[1:] * x_rec_con[:-1]) / (L_rec - 1)  # R(1)

        # Frequency search
        fre_num = 4096  # Number of frequency search points
        delta_f = Fs / 2 / fre_num  # Frequency search interval
        function_J_2 = np.zeros(int((Fs / 2 - delta_f) / delta_f) + 1, dtype=np.float64)  # Estimation function

        # Frequency search loop
        for idx, fre_find in enumerate(np.arange(delta_f, Fs / 2, delta_f)):
            function_J_2[idx] = np.real((R_1 * np.conj(R_0)) * np.exp(-1j * 2 * np.pi * fre_find / Fs)) ** 2

        # Maximum value corresponding to the center frequency
        J_max = np.max(function_J_2)
        idex_max = np.argmax(function_J_2)
        fre_center_esti = idex_max * delta_f
        center_frequency_output = (Fs - 2 * fre_center_esti) * height / (2 * Fs) / 0.992
    else:
        center_frequency_output = center_frequency_b * 2 * 1.001 / 0.993 / 0.992

    # write_to_excel(row_num, height, bandwidth_output, center_frequency_output)
    return [height, center_frequency_output]
