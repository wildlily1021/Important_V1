% 简单版：快速去除signal_3.5_1.75_converted.txt的直流分量
% 为80岁奶奶编写

clear all; close all; clc;

%% 读取数据
fprintf('正在读取文件...\n');

fid = fopen('signal_3.5_1.75_converted.txt', 'r');
data = [];
count = 0;

while ~feof(fid) && count < 10000
    line = fgetl(fid);
    if ischar(line)
        line = strtrim(line);
        if ~isempty(line) && line(1) == '(' && line(end) == ')'
            content = line(2:end-1);
            j_pos = strfind(content, 'j');
            if ~isempty(j_pos)
                real_imag = content(1:j_pos-1);
                
                if contains(real_imag, '+')
                    parts = strsplit(real_imag, '+');
                    real_part = str2double(parts{1});
                    imag_part = str2double(parts{2});
                elseif contains(real_imag, '-') && ~strcmp(real_imag(1), '-')
                    idx = strfind(real_imag, '-');
                    if length(idx) > 1
                        real_part = str2double(real_imag(1:idx(2)-1));
                        imag_part = -str2double(real_imag(idx(2)+1:end));
                    else
                        real_part = str2double(real_imag);
                        imag_part = 0;
                    end
                else
                    real_part = str2double(real_imag);
                    imag_part = 0;
                end
                
                if ~isnan(real_part) && ~isnan(imag_part)
                    data = [data; complex(real_part, imag_part)];
                    count = count + 1;
                end
            end
        end
    end
end
fclose(fid);

fprintf('读取完成！共 %d 个数据点\n', length(data));

%% 计算直流分量
dc_component = mean(data);
fprintf('\n直流分量：%.6f%+.6fj\n', real(dc_component), imag(dc_component));

%% 去除直流分量
data_no_dc = data - dc_component;
fprintf('去除直流分量后：%.6f%+.6fj\n', mean(real(data_no_dc)), mean(imag(data_no_dc)));

%% 保存结果
output_file = 'signal_3.5_1.75_no_dc.txt';
fid = fopen(output_file, 'w');
for i = 1:length(data_no_dc)
    fprintf(fid, '%.8f%+.8fj\n', real(data_no_dc(i)), imag(data_no_dc(i)));
end
fclose(fid);

fprintf('\n结果已保存到：%s\n', output_file);

%% 简单对比图
figure('Position', [100, 100, 800, 400]);

subplot(1,2,1);
plot_points = min(500, length(data));
plot(real(data(1:plot_points)), 'b-');
hold on;
plot(imag(data(1:plot_points)), 'r-');
title('原始信号');
xlabel('采样点');
ylabel('幅度');
legend('实部', '虚部');
grid on;

subplot(1,2,2);
plot(real(data_no_dc(1:plot_points)), 'b-');
hold on;
plot(imag(data_no_dc(1:plot_points)), 'r-');
title('去除直流分量后');
xlabel('采样点');
ylabel('幅度');
legend('实部', '虚部');
grid on;

fprintf('\n处理完成！\n');
