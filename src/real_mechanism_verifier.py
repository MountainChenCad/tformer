"""
真实机理验证系统 - 基于实际振动信号的物理机理验证
对目标域数据进行真实的频域分析和故障特征频率检测
"""

import os
import numpy as np
import scipy.io as sio
import scipy.signal as signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path

class RealMechanismVerifier:
    """真实机理验证器"""

    def __init__(self, sampling_rate=32000):
        self.sampling_rate = sampling_rate

        # 列车轴承参数 (基于实际工况)
        self.bearing_params = {
            'rpm': 600,           # 轴承转速约600rpm (90km/h时)
            'fr': 600 / 60,      # 转频 10Hz
            'Nd': 9,             # 滚动体数量 (假设类似SKF轴承)
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

    def perform_frequency_analysis(self, signal_data):
        """执行频域分析"""
        # 信号预处理
        signal_data = signal_data - np.mean(signal_data)  # 去直流

        # 应用窗函数
        from scipy.signal.windows import hann
        window = hann(len(signal_data))
        windowed_signal = signal_data * window

        # FFT分析
        fft_result = fft(windowed_signal)
        freqs = fftfreq(len(signal_data), 1/self.sampling_rate)

        # 取正频率部分
        positive_freq_mask = freqs >= 0
        freqs = freqs[positive_freq_mask]
        fft_magnitude = np.abs(fft_result[positive_freq_mask])

        # 转换为功率谱密度
        psd = fft_magnitude ** 2 / (self.sampling_rate * len(signal_data))

        # 转换为dB
        psd_db = 10 * np.log10(psd + 1e-12)  # 避免log(0)

        return freqs, psd_db

    def detect_fault_frequency(self, freqs, psd_db, target_freq, tolerance_pct=10):
        """检测特定故障特征频率"""
        if target_freq is None:
            return {
                'detected': False,
                'peak_freq': None,
                'peak_magnitude': None,
                'prominence': None,
                'snr': None
            }

        # 确定搜索范围
        tolerance = target_freq * tolerance_pct / 100
        freq_min = target_freq - tolerance
        freq_max = target_freq + tolerance

        # 在目标频率范围内搜索
        freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
        if not np.any(freq_mask):
            return {
                'detected': False,
                'peak_freq': None,
                'peak_magnitude': None,
                'prominence': None,
                'snr': None
            }

        search_freqs = freqs[freq_mask]
        search_psd = psd_db[freq_mask]

        # 寻找峰值
        peaks, properties = signal.find_peaks(search_psd, prominence=1.0, height=-np.inf)

        if len(peaks) == 0:
            # 没找到明显峰值，取最大值点
            max_idx = np.argmax(search_psd)
            peak_freq = search_freqs[max_idx]
            peak_magnitude = search_psd[max_idx]
            prominence = 0
        else:
            # 取最高的峰值
            highest_peak_idx = peaks[np.argmax(properties['peak_heights'])]
            peak_freq = search_freqs[highest_peak_idx]
            peak_magnitude = search_psd[highest_peak_idx]
            prominence = properties['prominences'][np.argmax(properties['peak_heights'])]

        # 计算信噪比 (峰值与周围噪声的比值)
        # 选择峰值周围±2Hz范围作为噪声参考
        noise_range = 2.0  # Hz
        noise_mask = (freqs >= peak_freq - noise_range) & (freqs <= peak_freq + noise_range)
        noise_mask = noise_mask & ~freq_mask  # 排除峰值区域

        if np.any(noise_mask):
            noise_level = np.mean(psd_db[noise_mask])
            snr = peak_magnitude - noise_level
        else:
            # 使用全频谱平均作为噪声基准
            noise_level = np.mean(psd_db)
            snr = peak_magnitude - noise_level

        # 判断是否检测到明显的故障频率
        detected = (prominence > 3.0) or (snr > 6.0)  # 突出度>3dB 或 信噪比>6dB

        return {
            'detected': bool(detected),  # 确保为Python bool类型
            'peak_freq': float(peak_freq),
            'peak_magnitude': float(peak_magnitude),
            'prominence': float(prominence),
            'snr': float(snr),
            'target_freq': float(target_freq) if target_freq else None,
            'search_range': [float(freq_min), float(freq_max)]
        }

    def verify_diagnosis_hypothesis(self, file_path, predicted_fault_type):
        """验证诊断假设"""
        print(f"机理验证: {os.path.basename(file_path)} - 预测故障: {predicted_fault_type}")

        # 加载信号
        signal_data = self.load_target_signal(file_path)
        if signal_data is None:
            return self._create_failed_result("信号加载失败")

        # 频域分析
        freqs, psd_db = self.perform_frequency_analysis(signal_data)

        # 确定目标频率
        fault_freq_key = self.fault_type_mapping.get(predicted_fault_type)
        target_frequency = self.fault_frequencies.get(fault_freq_key) if fault_freq_key else None

        # 检测故障特征频率
        detection_result = self.detect_fault_frequency(freqs, psd_db, target_frequency)

        # 评估验证结果
        if predicted_fault_type == 'Normal':
            # 正常状态：检查是否存在明显的故障特征频率
            all_fault_detected = []
            for fault_key, freq in self.fault_frequencies.items():
                result = self.detect_fault_frequency(freqs, psd_db, freq)
                all_fault_detected.append(result['detected'])

            verification_success = not any(all_fault_detected)  # 正常状态不应检测到故障频率
            confidence = "high" if verification_success else "low"
        else:
            # 故障状态：检查对应的故障特征频率
            verification_success = detection_result['detected']
            if verification_success:
                if detection_result['prominence'] > 6.0 and detection_result['snr'] > 10.0:
                    confidence = "high"
                elif detection_result['prominence'] > 3.0 and detection_result['snr'] > 6.0:
                    confidence = "medium"
                else:
                    confidence = "low"
            else:
                confidence = "failed"

        # 构建验证报告
        verification_report = {
            'file_name': os.path.basename(file_path),
            'predicted_fault': predicted_fault_type,
            'target_frequency_hz': float(target_frequency) if target_frequency else None,
            'detection_result': detection_result,
            'verification_success': bool(verification_success),
            'confidence_level': confidence,
            'analysis_metadata': {
                'signal_length': int(len(signal_data)),
                'sampling_rate': int(self.sampling_rate),
                'frequency_resolution': float(self.sampling_rate / len(signal_data)),
                'max_frequency': float(self.sampling_rate / 2)
            }
        }

        print(f"  结果: {'成功' if verification_success else '失败'} (置信度: {confidence})")
        if detection_result['detected']:
            if target_frequency:
                print(f"  检测频率: {detection_result['peak_freq']:.2f}Hz (目标: {target_frequency:.2f}Hz)")
            else:
                print(f"  检测频率: {detection_result['peak_freq']:.2f}Hz")
            print(f"  突出度: {detection_result['prominence']:.2f}dB, 信噪比: {detection_result['snr']:.2f}dB")

        return verification_report

    def _create_failed_result(self, error_message):
        """创建失败结果"""
        return {
            'verification_success': bool(False),
            'confidence_level': 'failed',
            'error_message': str(error_message),
            'detection_result': {
                'detected': bool(False),
                'peak_freq': None,
                'peak_magnitude': None,
                'prominence': None,
                'snr': None
            }
        }

    def batch_verify_target_domain(self, target_data_path, predictions_csv_path):
        """批量验证目标域文件"""
        print("\n=== 开始批量机理验证 ===")

        # 加载预测结果
        predictions_df = pd.read_csv(predictions_csv_path)

        verification_results = []

        for _, row in predictions_df.iterrows():
            file_name = row['File_Name']
            predicted_fault = row['Predicted_Fault_EN']

            file_path = os.path.join(target_data_path, file_name)

            if os.path.exists(file_path):
                result = self.verify_diagnosis_hypothesis(file_path, predicted_fault)
                verification_results.append(result)
            else:
                print(f"⚠ 文件不存在: {file_path}")
                result = self._create_failed_result(f"文件不存在: {file_name}")
                result['file_name'] = file_name
                result['predicted_fault'] = predicted_fault
                verification_results.append(result)

        return verification_results

    def generate_verification_report(self, verification_results, output_path):
        """生成验证报告"""

        # 统计分析
        total_files = len(verification_results)
        successful_verifications = sum(1 for r in verification_results if r['verification_success'])
        high_confidence = sum(1 for r in verification_results if r.get('confidence_level') == 'high')
        medium_confidence = sum(1 for r in verification_results if r.get('confidence_level') == 'medium')
        low_confidence = sum(1 for r in verification_results if r.get('confidence_level') == 'low')
        failed_verifications = sum(1 for r in verification_results if r.get('confidence_level') == 'failed')

        # 创建详细报告
        report = {
            'summary': {
                'total_files': int(total_files),
                'successful_verifications': int(successful_verifications),
                'success_rate': float(successful_verifications / total_files * 100 if total_files > 0 else 0),
                'confidence_distribution': {
                    'high': int(high_confidence),
                    'medium': int(medium_confidence),
                    'low': int(low_confidence),
                    'failed': int(failed_verifications)
                }
            },
            'bearing_parameters': self.bearing_params,
            'theoretical_frequencies': self.fault_frequencies,
            'detailed_results': verification_results
        }

        # 保存JSON报告
        json_path = os.path.join(output_path, 'mechanism_verification_report.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 创建CSV摘要
        csv_data = []
        for result in verification_results:
            csv_row = {
                'File_Name': result.get('file_name', 'Unknown'),
                'Predicted_Fault': result.get('predicted_fault', 'Unknown'),
                'Target_Frequency_Hz': result.get('target_frequency_hz', 'N/A'),
                'Verification_Success': result['verification_success'],
                'Confidence_Level': result.get('confidence_level', 'Unknown'),
                'Detected_Frequency_Hz': result['detection_result'].get('peak_freq', 'N/A'),
                'Peak_Magnitude_dB': result['detection_result'].get('peak_magnitude', 'N/A'),
                'Prominence_dB': result['detection_result'].get('prominence', 'N/A'),
                'SNR_dB': result['detection_result'].get('snr', 'N/A')
            }
            csv_data.append(csv_row)

        csv_df = pd.DataFrame(csv_data)
        csv_path = os.path.join(output_path, 'mechanism_verification_summary.csv')
        csv_df.to_csv(csv_path, index=False)

        print(f"\n=== 机理验证报告生成完成 ===")
        print(f"总文件数: {total_files}")
        print(f"验证成功: {successful_verifications} ({successful_verifications/total_files*100:.1f}%)")
        print(f"置信度分布: 高({high_confidence}) 中({medium_confidence}) 低({low_confidence}) 失败({failed_verifications})")
        print(f"详细报告: {json_path}")
        print(f"摘要CSV: {csv_path}")

        return report

def main():
    """主函数"""
    print("启动真实机理验证系统...")

    # 路径配置
    base_path = '/root/projects/tformer'
    target_data_path = os.path.join(base_path, 'data/target_domain')
    predictions_csv_path = os.path.join(base_path, 'results/target_domain_predictions_all_channels.csv')
    output_path = os.path.join(base_path, 'results')

    # 创建验证器
    verifier = RealMechanismVerifier(sampling_rate=32000)

    # 批量验证
    verification_results = verifier.batch_verify_target_domain(target_data_path, predictions_csv_path)

    # 生成报告
    report = verifier.generate_verification_report(verification_results, output_path)

    return report

if __name__ == "__main__":
    main()