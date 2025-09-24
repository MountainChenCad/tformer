"""
包络谱分析机理验证系统 - 基于希尔伯特变换的包络谱故障诊断
结合LLM推理引擎进行智能故障分析和决策
"""

import os
import numpy as np
import scipy.io as sio
import scipy.signal as signal
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class EnvelopeSpectrumVerifier:
    """基于包络谱分析的机理验证器"""

    def __init__(self, sampling_rate=32000):
        self.sampling_rate = sampling_rate

        # 列车轴承参数 (基于实际工况)
        self.bearing_params = {
            'rpm': 600,           # 轴承转速约600rpm (90km/h时)
            'fr': 600 / 60,      # 转频 10Hz
            'Nd': 9,             # 滚动体数量
            'd': 0.008,          # 滚动体直径 (m)
            'D': 0.04,           # 轴承节径 (m)
        }

        # 计算理论故障特征频率
        self.fault_frequencies = self._calculate_fault_frequencies()

        # 故障类型映射
        self.fault_type_mapping = {
            'Normal': None,
            'Inner': 'BPFI',
            'Outer': 'BPFO',
            'Ball': 'BSF'
        }

        print(f"理论故障特征频率:")
        for fault, freq in self.fault_frequencies.items():
            print(f"  {fault}: {freq:.2f} Hz")

    def _calculate_fault_frequencies(self):
        """计算理论故障特征频率"""
        params = self.bearing_params
        fr = params['fr']
        Nd = params['Nd']
        d = params['d']
        D = params['D']

        frequencies = {
            'BPFI': fr * Nd/2 * (1 + d/D),    # 内圈故障特征频率 ≈ 54Hz
            'BPFO': fr * Nd/2 * (1 - d/D),    # 外圈故障特征频率 ≈ 36Hz
            'BSF': fr * D/(2*d) * (1 - (d/D)**2),  # 滚动体故障特征频率 ≈ 24Hz
            'FTF': fr/2 * (1 - d/D),          # 滚动体公转频率 ≈ 4Hz
        }

        return frequencies

    def load_target_signal(self, file_path):
        """加载目标域振动信号"""
        try:
            mat_data = sio.loadmat(file_path)

            # 寻找振动信号数据
            signal_data = None
            signal_key = None

            for key in mat_data.keys():
                if not key.startswith('__') and isinstance(mat_data[key], np.ndarray):
                    data = mat_data[key].flatten()
                    if len(data) > 1000:  # 找到较长的数据向量
                        signal_data = data
                        signal_key = key
                        break

            if signal_data is None:
                raise ValueError("未找到有效的振动信号数据")

            print(f"  加载信号: {signal_key}, 长度: {len(signal_data)}, 采样率: {self.sampling_rate}Hz")

            return signal_data

        except Exception as e:
            print(f"  ⚠ 加载文件失败: {e}")
            return None

    def compute_envelope_spectrum(self, signal_data):
        """计算包络谱"""
        # 信号预处理
        signal_data = signal_data - np.mean(signal_data)  # 去直流

        # 高通滤波去除低频干扰
        sos = signal.butter(4, 50, btype='highpass', fs=self.sampling_rate, output='sos')
        filtered_signal = signal.sosfilt(sos, signal_data)

        # 带通滤波提取共振频段 (1000-8000Hz)
        sos_bp = signal.butter(4, [1000, 8000], btype='bandpass', fs=self.sampling_rate, output='sos')
        bandpass_signal = signal.sosfilt(sos_bp, filtered_signal)

        # 希尔伯特变换获取解析信号
        analytic_signal = hilbert(bandpass_signal)

        # 计算包络信号
        envelope = np.abs(analytic_signal)

        # 去除包络信号的直流分量
        envelope = envelope - np.mean(envelope)

        # 计算包络谱（包络信号的FFT）
        envelope_fft = fft(envelope)
        envelope_freqs = fftfreq(len(envelope), 1/self.sampling_rate)

        # 取正频率部分
        positive_freq_mask = envelope_freqs >= 0
        envelope_freqs = envelope_freqs[positive_freq_mask]
        envelope_magnitude = np.abs(envelope_fft[positive_freq_mask])

        # 转换为dB
        envelope_spectrum_db = 20 * np.log10(envelope_magnitude + 1e-12)

        return envelope_freqs, envelope_spectrum_db, envelope

    def detect_envelope_peaks(self, envelope_freqs, envelope_spectrum_db, target_freq, tolerance_pct=15):
        """检测包络谱中的故障特征频率峰值"""
        if target_freq is None:
            return {
                'detected': False,
                'peak_freq': None,
                'peak_magnitude': None,
                'prominence': None,
                'snr': None,
                'harmonics': []
            }

        # 确定搜索范围
        tolerance = target_freq * tolerance_pct / 100
        freq_min = target_freq - tolerance
        freq_max = target_freq + tolerance

        # 在目标频率范围内搜索
        freq_mask = (envelope_freqs >= freq_min) & (envelope_freqs <= freq_max)
        if not np.any(freq_mask):
            return {
                'detected': False,
                'peak_freq': None,
                'peak_magnitude': None,
                'prominence': None,
                'snr': None,
                'harmonics': []
            }

        search_freqs = envelope_freqs[freq_mask]
        search_spectrum = envelope_spectrum_db[freq_mask]

        # 寻找峰值
        peaks, properties = signal.find_peaks(search_spectrum, prominence=2.0, height=-np.inf)

        if len(peaks) == 0:
            # 没找到明显峰值，取最大值点
            max_idx = np.argmax(search_spectrum)
            peak_freq = search_freqs[max_idx]
            peak_magnitude = search_spectrum[max_idx]
            prominence = 0
        else:
            # 取最高的峰值
            highest_peak_idx = peaks[np.argmax(properties['peak_heights'])]
            peak_freq = search_freqs[highest_peak_idx]
            peak_magnitude = search_spectrum[highest_peak_idx]
            prominence = properties['prominences'][np.argmax(properties['peak_heights'])]

        # 计算信噪比
        noise_range = 5.0  # Hz
        noise_mask = (envelope_freqs >= peak_freq - noise_range) & (envelope_freqs <= peak_freq + noise_range)
        noise_mask = noise_mask & ~freq_mask  # 排除峰值区域

        if np.any(noise_mask):
            noise_level = np.mean(envelope_spectrum_db[noise_mask])
            snr = peak_magnitude - noise_level
        else:
            # 使用全频谱平均作为噪声基准
            noise_level = np.mean(envelope_spectrum_db)
            snr = peak_magnitude - noise_level

        # 检测谐波
        harmonics = []
        for i in range(2, 6):  # 检测2-5次谐波
            harmonic_freq = target_freq * i
            if harmonic_freq < envelope_freqs[-1]:
                harmonic_tolerance = harmonic_freq * 0.1
                harmonic_mask = (envelope_freqs >= harmonic_freq - harmonic_tolerance) & \
                              (envelope_freqs <= harmonic_freq + harmonic_tolerance)
                if np.any(harmonic_mask):
                    harmonic_spectrum = envelope_spectrum_db[harmonic_mask]
                    harmonic_peak_idx = np.argmax(harmonic_spectrum)
                    harmonic_peak_freq = envelope_freqs[harmonic_mask][harmonic_peak_idx]
                    harmonic_magnitude = harmonic_spectrum[harmonic_peak_idx]

                    if harmonic_magnitude > noise_level + 3:  # 谐波显著性判断
                        harmonics.append({
                            'order': i,
                            'freq': float(harmonic_peak_freq),
                            'magnitude': float(harmonic_magnitude)
                        })

        # 判断是否检测到明显的故障频率
        detected = (prominence > 5.0) or (snr > 8.0)  # 包络谱需要更高阈值

        return {
            'detected': bool(detected),
            'peak_freq': float(peak_freq),
            'peak_magnitude': float(peak_magnitude),
            'prominence': float(prominence),
            'snr': float(snr),
            'target_freq': float(target_freq),
            'search_range': [float(freq_min), float(freq_max)],
            'harmonics': harmonics
        }

    def analyze_envelope_spectrum(self, file_path, predicted_fault_type):
        """完整的包络谱分析流程"""
        print(f"\n包络谱分析: {os.path.basename(file_path)} - 预测故障: {predicted_fault_type}")

        # 加载信号
        signal_data = self.load_target_signal(file_path)
        if signal_data is None:
            return self._create_failed_result("信号加载失败")

        # 计算包络谱
        envelope_freqs, envelope_spectrum_db, envelope_signal = self.compute_envelope_spectrum(signal_data)

        # 确定目标频率
        fault_freq_key = self.fault_type_mapping.get(predicted_fault_type)
        target_frequency = self.fault_frequencies.get(fault_freq_key) if fault_freq_key else None

        # 检测故障特征频率
        detection_result = self.detect_envelope_peaks(envelope_freqs, envelope_spectrum_db, target_frequency)

        # 评估验证结果
        if predicted_fault_type == 'Normal':
            # 正常状态：检查是否存在明显的故障特征频率
            all_fault_detected = []
            for fault_key, freq in self.fault_frequencies.items():
                result = self.detect_envelope_peaks(envelope_freqs, envelope_spectrum_db, freq)
                all_fault_detected.append(result['detected'])

            verification_success = not any(all_fault_detected)  # 正常状态不应检测到故障频率
            confidence = "high" if verification_success else "low"
        else:
            # 故障状态：检查对应的故障特征频率
            verification_success = detection_result['detected']
            if verification_success:
                harmonic_bonus = len(detection_result['harmonics']) * 0.1
                adjusted_prominence = detection_result['prominence'] + harmonic_bonus * 5
                adjusted_snr = detection_result['snr'] + harmonic_bonus * 3

                if adjusted_prominence > 10.0 and adjusted_snr > 12.0:
                    confidence = "very_high"
                elif detection_result['prominence'] > 8.0 and detection_result['snr'] > 10.0:
                    confidence = "high"
                elif detection_result['prominence'] > 5.0 and detection_result['snr'] > 8.0:
                    confidence = "medium"
                else:
                    confidence = "low"
            else:
                confidence = "failed"

        # 生成LLM分析文本
        llm_analysis_text = self._generate_llm_analysis_text(
            file_path, predicted_fault_type, target_frequency, detection_result,
            envelope_freqs, envelope_spectrum_db
        )

        # 构建完整分析报告
        analysis_report = {
            'file_name': os.path.basename(file_path),
            'predicted_fault': predicted_fault_type,
            'target_frequency_hz': float(target_frequency) if target_frequency else None,
            'envelope_detection_result': detection_result,
            'verification_success': bool(verification_success),
            'confidence_level': confidence,
            'llm_analysis': llm_analysis_text,
            'envelope_spectrum_data': {
                'frequencies': envelope_freqs[:1000].tolist(),  # 限制数据量
                'spectrum_db': envelope_spectrum_db[:1000].tolist(),
                'envelope_signal_sample': envelope_signal[:5000].tolist() if len(envelope_signal) > 5000 else envelope_signal.tolist()
            },
            'analysis_metadata': {
                'signal_length': int(len(signal_data)),
                'sampling_rate': int(self.sampling_rate),
                'envelope_freq_resolution': float(self.sampling_rate / len(signal_data)),
                'analysis_method': 'Hilbert Transform + Envelope Spectrum'
            }
        }

        print(f"  包络谱结果: {'成功' if verification_success else '失败'} (置信度: {confidence})")
        if detection_result['detected']:
            if target_frequency:
                print(f"  检测频率: {detection_result['peak_freq']:.2f}Hz (目标: {target_frequency:.2f}Hz)")
            print(f"  突出度: {detection_result['prominence']:.2f}dB, 信噪比: {detection_result['snr']:.2f}dB")
            if detection_result['harmonics']:
                print(f"  检测到{len(detection_result['harmonics'])}个谐波分量")

        return analysis_report

    def _generate_llm_analysis_text(self, file_path, predicted_fault_type, target_frequency,
                                   detection_result, envelope_freqs, envelope_spectrum_db):
        """为LLM生成详细的分析文本"""

        file_name = os.path.basename(file_path)

        # 构建基础信息
        analysis_lines = [
            f"文件名: {file_name}",
            f"PATE模型预测故障类型: {predicted_fault_type}",
            f"理论故障特征频率: {target_frequency:.2f}Hz" if target_frequency else "理论故障特征频率: 无(正常状态)",
            "",
            "包络谱分析结果:",
        ]

        if detection_result['detected']:
            analysis_lines.extend([
                f"- 检测到故障特征频率: {detection_result['peak_freq']:.2f}Hz",
                f"- 频率偏差: {abs(detection_result['peak_freq'] - target_frequency):.2f}Hz ({abs(detection_result['peak_freq'] - target_frequency)/target_frequency*100:.1f}%)" if target_frequency else "",
                f"- 峰值幅度: {detection_result['peak_magnitude']:.2f}dB",
                f"- 峰值突出度: {detection_result['prominence']:.2f}dB",
                f"- 信噪比: {detection_result['snr']:.2f}dB"
            ])

            if detection_result['harmonics']:
                analysis_lines.append(f"- 检测到{len(detection_result['harmonics'])}个谐波分量:")
                for harmonic in detection_result['harmonics']:
                    analysis_lines.append(f"  * {harmonic['order']}次谐波: {harmonic['freq']:.2f}Hz ({harmonic['magnitude']:.2f}dB)")
        else:
            analysis_lines.extend([
                "- 未检测到明显的故障特征频率",
                f"- 在目标频率{target_frequency:.2f}Hz附近未发现显著峰值" if target_frequency else ""
            ])

        # 添加频谱统计信息
        spectrum_stats = {
            'max_amplitude': float(np.max(envelope_spectrum_db)),
            'mean_amplitude': float(np.mean(envelope_spectrum_db)),
            'std_amplitude': float(np.std(envelope_spectrum_db))
        }

        analysis_lines.extend([
            "",
            "包络谱统计特征:",
            f"- 最大幅度: {spectrum_stats['max_amplitude']:.2f}dB",
            f"- 平均幅度: {spectrum_stats['mean_amplitude']:.2f}dB",
            f"- 幅度标准差: {spectrum_stats['std_amplitude']:.2f}dB"
        ])

        return "\n".join([line for line in analysis_lines if line is not None])

    def _create_failed_result(self, error_message):
        """创建失败结果"""
        return {
            'verification_success': bool(False),
            'confidence_level': 'failed',
            'error_message': str(error_message),
            'envelope_detection_result': {
                'detected': bool(False),
                'peak_freq': None,
                'peak_magnitude': None,
                'prominence': None,
                'snr': None,
                'harmonics': []
            },
            'llm_analysis': f"分析失败: {error_message}"
        }

def main():
    """主函数"""
    print("启动包络谱分析机理验证系统...")

    # 路径配置
    base_path = '/root/projects/tformer'
    target_data_path = os.path.join(base_path, 'data/target_domain')
    predictions_csv_path = os.path.join(base_path, 'results/target_domain_predictions_all_channels.csv')
    output_path = os.path.join(base_path, 'results')

    # 创建验证器
    verifier = EnvelopeSpectrumVerifier(sampling_rate=32000)

    # 加载预测结果
    predictions_df = pd.read_csv(predictions_csv_path)

    envelope_results = []

    print("\n=== 开始批量包络谱分析 ===")
    for _, row in predictions_df.iterrows():
        file_name = row['File_Name']
        predicted_fault = row['Predicted_Fault_EN']

        file_path = os.path.join(target_data_path, file_name)

        if os.path.exists(file_path):
            result = verifier.analyze_envelope_spectrum(file_path, predicted_fault)
            envelope_results.append(result)
        else:
            print(f"⚠ 文件不存在: {file_path}")
            result = verifier._create_failed_result(f"文件不存在: {file_name}")
            result['file_name'] = file_name
            result['predicted_fault'] = predicted_fault
            envelope_results.append(result)

    # 保存结果
    json_path = os.path.join(output_path, 'envelope_spectrum_analysis_report.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(envelope_results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 包络谱分析完成! 结果保存至: {json_path}")

    return envelope_results

if __name__ == "__main__":
    main()