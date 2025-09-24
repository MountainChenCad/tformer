"""
机理验证模块 - 基于物理机理的自动化故障验证系统
实现基于诊断假设的定向物理机理验证，为闭环诊断提供第二重证据
"""

import numpy as np
import scipy.signal as signal
from scipy import fft
from scipy.signal import hilbert, find_peaks
import warnings
warnings.filterwarnings('ignore')

class BearingParameters:
    """轴承参数配置类"""

    def __init__(self):
        # 列车轴承参数 (基于题目描述)
        self.train_bearing = {
            'rpm': 600,           # 轴承转速 (题目给出约600rpm)
            'fr': 600 / 60,      # 转频 Hz
            'Nd': 9,             # 滚动体数量 (假设SKF类似参数)
            'd': 0.008,          # 滚动体直径 (m)
            'D': 0.04,           # 轴承节径 (m)
        }

    def calculate_fault_frequencies(self):
        """计算理论故障特征频率"""
        params = self.train_bearing
        fr = params['fr']
        Nd = params['Nd']
        d = params['d']
        D = params['D']

        # 计算故障特征频率
        frequencies = {
            'BPFI': fr * Nd/2 * (1 + d/D),    # 内圈故障特征频率
            'BPFO': fr * Nd/2 * (1 - d/D),    # 外圈故障特征频率
            'BSF': fr * D/(2*d) * (1 - (d/D)**2),  # 滚动体故障特征频率
            'FTF': fr/2 * (1 - d/D),          # 滚动体公转频率
        }

        return frequencies

class MechanismVerifier:
    """机理验证核心类"""

    def __init__(self, sampling_rate=32000):
        self.sampling_rate = sampling_rate
        self.bearing_params = BearingParameters()
        self.fault_frequencies = self.bearing_params.calculate_fault_frequencies()

        # 故障类型映射到理论频率
        self.fault_type_mapping = {
            'Normal': None,
            'Inner': 'BPFI',
            'Outer': 'BPFO',
            'Ball': 'BSF'
        }

    def verify_hypothesis(self, signal_data, hypothesis):
        """
        基于诊断假设进行定向机理验证

        Args:
            signal_data: 原始振动信号 numpy array
            hypothesis: PATE模型的诊断假设 dict {'fault_type': str, 'confidence': float}

        Returns:
            mechanism_evidence: 机理证据 dict
        """
        fault_type = hypothesis.get('fault_type', 'Normal')

        # 如果是正常状态，不进行机理验证
        if fault_type == 'Normal':
            return {
                'analysis_type': 'No_Analysis_Required',
                'target_fault_type': fault_type,
                'theoretical_frequency_hz': None,
                'peak_found': None,
                'peak_prominence_db': None,
                'harmonics_detected': 0,
                'verification_result': 'NORMAL_STATE'
            }

        # 获取目标故障的理论频率
        freq_key = self.fault_type_mapping.get(fault_type)
        if not freq_key:
            return self._create_error_evidence(f"Unknown fault type: {fault_type}")

        target_frequency = self.fault_frequencies[freq_key]

        # 执行包络谱分析
        envelope_spectrum, frequencies = self._envelope_spectrum_analysis(signal_data)

        # 在理论频率附近检测峰值
        evidence = self._detect_mechanism_evidence(
            envelope_spectrum, frequencies, target_frequency, fault_type
        )

        return evidence

    def _envelope_spectrum_analysis(self, signal_data):
        """
        包络谱分析：希尔伯特变换 + FFT

        Args:
            signal_data: 输入信号

        Returns:
            envelope_spectrum: 包络谱幅值
            frequencies: 对应的频率数组
        """
        # 1. 高通滤波去除低频成分
        nyquist = self.sampling_rate / 2
        high_cutoff = 1000  # 1kHz高通
        if high_cutoff < nyquist:
            sos = signal.butter(4, high_cutoff/nyquist, btype='high', output='sos')
            filtered_signal = signal.sosfilt(sos, signal_data)
        else:
            filtered_signal = signal_data

        # 2. 希尔伯特变换获取解析信号
        analytic_signal = hilbert(filtered_signal)

        # 3. 计算包络信号
        envelope = np.abs(analytic_signal)

        # 4. 去除直流分量
        envelope = envelope - np.mean(envelope)

        # 5. 对包络信号进行FFT
        n_fft = len(envelope)
        envelope_fft = fft.fft(envelope)
        envelope_spectrum = np.abs(envelope_fft[:n_fft//2])

        # 6. 频率轴
        frequencies = fft.fftfreq(n_fft, 1/self.sampling_rate)[:n_fft//2]

        return envelope_spectrum, frequencies

    def _detect_mechanism_evidence(self, envelope_spectrum, frequencies, target_frequency, fault_type):
        """
        在包络谱中检测目标频率处的峰值证据

        Args:
            envelope_spectrum: 包络谱数据
            frequencies: 频率数组
            target_frequency: 目标故障特征频率
            fault_type: 故障类型

        Returns:
            evidence: 机理证据字典
        """
        # 设置搜索范围 (±20%的理论频率)
        search_tolerance = 0.2
        freq_min = target_frequency * (1 - search_tolerance)
        freq_max = target_frequency * (1 + search_tolerance)

        # 找到搜索频率范围的索引
        search_indices = np.where((frequencies >= freq_min) & (frequencies <= freq_max))[0]

        if len(search_indices) == 0:
            return self._create_evidence(
                fault_type, target_frequency, False, 0.0, 0, 'NO_SEARCH_RANGE'
            )

        # 在搜索范围内寻找峰值
        search_spectrum = envelope_spectrum[search_indices]
        search_frequencies = frequencies[search_indices]

        # 峰值检测
        peaks, properties = find_peaks(
            search_spectrum,
            height=np.max(search_spectrum) * 0.1,  # 最小高度为最大值的10%
            distance=5,  # 峰值之间的最小距离
            prominence=np.std(search_spectrum) * 2  # 显著性阈值
        )

        if len(peaks) == 0:
            return self._create_evidence(
                fault_type, target_frequency, False, 0.0, 0, 'NO_PEAK_DETECTED'
            )

        # 找到最显著的峰值
        peak_heights = properties['peak_heights']
        max_peak_idx = np.argmax(peak_heights)
        peak_index = peaks[max_peak_idx]
        peak_frequency = search_frequencies[peak_index]
        peak_amplitude = peak_heights[max_peak_idx]

        # 计算峰值显著性 (相对于背景噪声的dB值)
        background_level = np.median(search_spectrum)
        if background_level > 0:
            prominence_db = 20 * np.log10(peak_amplitude / background_level)
        else:
            prominence_db = 0.0

        # 检测谐波
        harmonics_count = self._detect_harmonics(
            envelope_spectrum, frequencies, peak_frequency, prominence_db
        )

        # 判断是否发现有效峰值
        peak_found = prominence_db > 6.0  # 6dB以上认为是有效峰值

        # 确定验证结果
        if peak_found and harmonics_count >= 2:
            verification_result = 'STRONG_CONFIRMATION'
        elif peak_found:
            verification_result = 'WEAK_CONFIRMATION'
        else:
            verification_result = 'NO_CONFIRMATION'

        return self._create_evidence(
            fault_type, target_frequency, peak_found, prominence_db,
            harmonics_count, verification_result, peak_frequency
        )

    def _detect_harmonics(self, envelope_spectrum, frequencies, fundamental_freq, min_prominence_db):
        """检测谐波成分"""
        harmonics_count = 0
        max_harmonic = 5  # 检测前5次谐波

        for n in range(2, max_harmonic + 1):
            harmonic_freq = fundamental_freq * n

            # 搜索谐波频率附近
            harmonic_tolerance = fundamental_freq * 0.1  # 10%容差
            harmonic_indices = np.where(
                (frequencies >= harmonic_freq - harmonic_tolerance) &
                (frequencies <= harmonic_freq + harmonic_tolerance)
            )[0]

            if len(harmonic_indices) > 0:
                harmonic_spectrum = envelope_spectrum[harmonic_indices]
                if np.max(harmonic_spectrum) > 0:
                    harmonic_prominence = 20 * np.log10(
                        np.max(harmonic_spectrum) / np.median(envelope_spectrum)
                    )
                    if harmonic_prominence > min_prominence_db * 0.5:  # 谐波要求较低
                        harmonics_count += 1

        return harmonics_count

    def _create_evidence(self, fault_type, target_freq, peak_found, prominence_db,
                        harmonics_count, verification_result, detected_freq=None):
        """创建标准化的机理证据字典"""
        return {
            'analysis_type': 'Envelope_Spectrum_Analysis',
            'target_fault_type': fault_type,
            'theoretical_frequency_hz': round(target_freq, 2),
            'detected_frequency_hz': round(detected_freq, 2) if detected_freq else None,
            'peak_found': peak_found,
            'peak_prominence_db': round(prominence_db, 2),
            'harmonics_detected': harmonics_count,
            'verification_result': verification_result,
            'analysis_parameters': {
                'sampling_rate': self.sampling_rate,
                'search_tolerance': 0.2,
                'min_prominence_threshold': 6.0,
                'theoretical_frequencies': self.fault_frequencies
            }
        }

    def _create_error_evidence(self, error_message):
        """创建错误情况下的证据"""
        return {
            'analysis_type': 'Error',
            'error_message': error_message,
            'verification_result': 'ERROR'
        }

def test_mechanism_verifier():
    """测试函数"""
    print("Testing MechanismVerifier...")

    # 创建测试信号 (模拟内圈故障)
    fs = 32000
    t = np.linspace(0, 1, fs)

    # 基础噪声
    noise = np.random.normal(0, 0.1, len(t))

    # 模拟内圈故障 (BPFI ≈ 45Hz)
    bpfi = 45
    fault_signal = 0.3 * np.sin(2 * np.pi * bpfi * t) * np.exp(-5 * (t % (1/bpfi)))

    test_signal = noise + fault_signal

    # 创建验证器
    verifier = MechanismVerifier(sampling_rate=fs)

    # 测试假设
    hypothesis = {'fault_type': 'Inner', 'confidence': 0.9}

    # 执行验证
    evidence = verifier.verify_hypothesis(test_signal, hypothesis)

    print("Test completed!")
    print("Evidence:")
    for key, value in evidence.items():
        print(f"  {key}: {value}")

if __name__ == '__main__':
    test_mechanism_verifier()