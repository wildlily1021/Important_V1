import numpy as np
import re

def normalize_signal_data(input_file, output_file, num_rows=400000):
    """
    将信号数据归一化，从大数值转换为小数值格式
    """
    
    print(f"正在读取文件: {input_file}")
    
    # 读取数据
    data = []
    with open(input_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_rows:
                break
            
            line = line.strip()
            if line.startswith('(') and line.endswith('j)'):
                # 使用正则表达式解析复数
                # 匹配格式: (a+bj) 或 (a-bj) 或 (-a+bj) 或 (-a-bj)
                pattern = r'\(([+-]?\d+\.?\d*)([+-]\d+\.?\d*)j\)'
                match = re.match(pattern, line)
                
                if match:
                    real_part = float(match.group(1))
                    imag_part = float(match.group(2))
                    data.append(complex(real_part, imag_part))
                else:
                    print(f"无法解析的行: {line}")
    
    # 转换为numpy数组
    complex_data = np.array(data)
    
    print(f"读取了 {len(complex_data)} 个数据点")
    print(f"原始数据范围:")
    print(f"  实部: {np.min(complex_data.real):.2f} 到 {np.max(complex_data.rm eal):.2f}")
    print(f"  虚部: {np.min(complex_data.imag):.2f} 到 {np.max(complex_data.imag):.2f}")
    
    # 归一化方法1: 按最大值归一化
    max_abs = np.max(np.abs(complex_data))
    normalized_data = complex_data / max_abs
    
    print(f"归一化后数据范围:")
    print(f"  实部: {np.min(normalized_data.real):.8f} 到 {np.max(normalized_data.real):.8f}")
    print(f"  虚部: {np.min(normalized_data.imag):.8f} 到 {np.max(normalized_data.imag):.8f}")
    
    # 保存归一化后的数据
    print(f"正在保存到: {output_file}")
    np.savetxt(output_file, normalized_data, fmt='%.8f%+.8fj')
    
    print("转换完成！")
    print(f"输出文件: {output_file}")
    print(f"数据已归一化，数值范围在 -1 到 1 之间")

def normalize_by_std(input_file, output_file, num_rows=10000):
    """
    使用标准差归一化
    """
    print(f"正在读取文件: {input_file}")
    
    # 读取数据
    data = []
    with open(input_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_rows:
                break
            
            line = line.strip()
            if line.startswith('(') and line.endswith('j)'):
                # 使用正则表达式解析复数
                pattern = r'\(([+-]?\d+\.?\d*)([+-]\d+\.?\d*)j\)'
                match = re.match(pattern, line)
                
                if match:
                    real_part = float(match.group(1))
                    imag_part = float(match.group(2))
                    data.append(complex(real_part, imag_part))
    
    complex_data = np.array(data)
    
    # 使用标准差归一化
    std_real = np.std(complex_data.real)
    std_imag = np.std(complex_data.imag)
    
    normalized_data = complex(complex_data.real / std_real, complex_data.imag / std_imag)
    
    print(f"标准差归一化后数据范围:")
    print(f"  实部: {np.min(normalized_data.real):.8f} 到 {np.max(normalized_data.real):.8f}")
    print(f"  虚部: {np.min(normalized_data.imag):.8f} 到 {np.max(normalized_data.imag):.8f}")
    
    # 保存数据
    np.savetxt(output_file, normalized_data, fmt='%.8f%+.8fj')
    print(f"标准差归一化完成，保存到: {output_file}")

if __name__ == "__main__":
    # 设置文件路径
    input_file = "signal_3.5_1.75_converted.txt"
    output_file_max = "signal_3.5_1.75_normalized_max.txt"
    
    print("=== 方法1: 按最大值归一化 ===")
    normalize_signal_data(input_file, output_file_max, num_rows=10000)

    print("\n=== 转换完成 ===")
