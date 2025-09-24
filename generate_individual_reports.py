"""
为每个目标域序列生成详细的闭环验证诊断报告
基于三阶段验证架构：Hypothesize → Verify → Cross-Examine & Reason
"""

import json
import os
from datetime import datetime

def load_closed_loop_results():
    """加载闭环诊断结果"""
    with open('results/closed_loop_diagnosis_detailed_20250924_114125.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_individual_report(file_result, output_dir='results/individual_reports'):
    """为单个文件生成详细报告"""

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    file_name = file_result['file_name']
    final_diagnosis = file_result['final_diagnosis']
    confidence_level = file_result['confidence_level']

    # 获取代表性段的详细信息
    representative_segment = file_result['segment_results'][0]
    evidence_chain = representative_segment['evidence_chain']

    # PATE证据
    pate_evidence = evidence_chain['step1_pate_diagnosis']
    # 机理证据
    mech_evidence = evidence_chain['step2_mechanism_verification']
    # LLM推理
    llm_reasoning = evidence_chain['step3_llm_reasoning']

    # 故障类型中英文映射
    fault_mapping = {
        'Normal': '正常状态',
        'Inner': '内圈故障',
        'Outer': '外圈故障',
        'Ball': '滚动体故障'
    }

    # 理论频率映射
    freq_mapping = {
        'Inner': ('内圈故障特征频率 (BPFI)', 54.0),
        'Outer': ('外圈故障特征频率 (BPFO)', 36.0),
        'Ball': ('滚动体故障特征频率 (BSF)', 24.0)
    }

    report_content = f"""# 闭环验证诊断报告 - {file_name}

## 📋 诊断概览

**文件编号**: {file_name}
**最终诊断**: {final_diagnosis} ({fault_mapping.get(final_diagnosis, final_diagnosis)})
**系统置信度**: {confidence_level}
**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**分析段数**: {file_result['segment_count']}个信号段
**投票一致率**: {file_result['vote_ratio']:.3f} ({file_result['vote_ratio']*100:.1f}%)

## 🔄 三阶段闭环验证详细过程

### 阶段1: 初步诊断假设生成 (Hypothesize)
**执行模块**: PATE-Net深度学习模型

**诊断假设**:
- **预测故障类型**: {pate_evidence['fault_type']} ({pate_evidence['fault_type_cn']})
- **模型置信度**: {pate_evidence['confidence']:.4f} ({pate_evidence['confidence']*100:.2f}%)
- **原型距离**: {pate_evidence['distance_to_prototype']:.4f}
- **模型类型**: {pate_evidence['model_type']}

**类别概率分布**:
"""

    # 添加概率分布
    probs = pate_evidence['class_probabilities']
    fault_names = ['正常', '内圈故障', '外圈故障', '滚动体故障']
    for i, (name, prob) in enumerate(zip(fault_names, probs)):
        report_content += f"- {name}: {prob:.4f} ({prob*100:.2f}%)\n"

    report_content += f"""

**模型判断依据**:
模型通过分析振动信号的深层特征模式，认为该信号与"{pate_evidence['fault_type_cn']}"类别的标准原型最为相似，原型距离为{pate_evidence['distance_to_prototype']:.4f}。

### 阶段2: 物理机理定向验证 (Verify)
**执行模块**: MechanismVerifier机理验证器

**验证参数**:
- **验证目标**: {mech_evidence['target_fault_type']}故障
- **分析方法**: {mech_evidence['analysis_type']}
"""

    if mech_evidence['target_fault_type'] != 'Normal':
        freq_name, freq_value = freq_mapping.get(mech_evidence['target_fault_type'], ('未知频率', 0))
        report_content += f"""- **理论特征频率**: {freq_name} = {mech_evidence['theoretical_frequency_hz']} Hz
- **检测频率**: {mech_evidence.get('detected_frequency_hz', 'N/A')} Hz

**分析结果**:
- **峰值检测**: {'✓ 发现显著峰值' if mech_evidence['peak_found'] else '✗ 未发现显著峰值'}
- **峰值显著性**: {mech_evidence['peak_prominence_db']} dB
- **谐波检测**: {mech_evidence['harmonics_detected']}个谐波成分
- **验证结论**: {mech_evidence['verification_result']}

**机理分析说明**:
系统基于轴承故障机理，针对"{mech_evidence['target_fault_type']}"假设进行定向包络谱分析。通过希尔伯特变换提取包络信号，然后在{mech_evidence['theoretical_frequency_hz']} Hz理论频率附近搜索故障特征。"""

        # 根据验证结果添加详细说明
        if mech_evidence['verification_result'] == 'STRONG_CONFIRMATION':
            report_content += f"\n结果显示在理论频率处发现强烈峰值（{mech_evidence['peak_prominence_db']} dB），并检测到{mech_evidence['harmonics_detected']}个谐波，强烈支持故障假设。"
        elif mech_evidence['verification_result'] == 'WEAK_CONFIRMATION':
            report_content += f"\n结果显示在理论频率处发现较弱峰值（{mech_evidence['peak_prominence_db']} dB），部分支持故障假设，但证据强度有限。"
        elif mech_evidence['verification_result'] == 'NO_CONFIRMATION':
            report_content += f"\n结果显示在理论频率处未发现显著峰值，机理验证不支持当前故障假设。"
        else:
            report_content += f"\n分析过程中遇到技术问题，无法完成有效的机理验证。"
    else:
        report_content += f"""
**正常状态验证**: 由于PATE模型诊断为正常状态，无需进行故障特征频率验证。"""

    report_content += f"""

### 阶段3: LLM专家会诊推理 (Cross-Examine & Reason)
**执行模块**: LLMReasoningEngine专家推理引擎

**专家角色**: 世界顶级机械故障诊断专家
**推理任务**: 基于双重证据进行逻辑推理和一致性判断

**证据整合分析**:
- **证据一致性**: {llm_reasoning['evidence_consistency']}
- **推理结论**: {llm_reasoning['final_conclusion']}
- **专家置信度**: {llm_reasoning['confidence_level']}

**专家推理过程**:
{llm_reasoning['reasoning_process']}

## 📊 综合诊断质量评估

**多维度质量指标**:
"""

    # 添加质量评估
    quality = representative_segment['diagnosis_quality']
    report_content += f"""- **PATE模型置信度**: {quality['pate_confidence']:.4f}
- **机理验证结果**: {quality['mechanism_confirmation']}
- **证据一致性评估**: {quality['evidence_agreement']}
- **系统整体可靠性**: {quality['overall_reliability']}

**可靠性等级说明**:
"""

    reliability = quality['overall_reliability']
    if reliability == 'VERY_HIGH':
        report_content += "🟢 **极高可靠性** - PATE高置信度(>0.9) + 机理强确认 + LLM高置信度，三重证据完美闭环"
    elif reliability == 'HIGH':
        report_content += "🔵 **高可靠性** - PATE高置信度(>0.8) + 机理确认，双重证据相互支撑"
    elif reliability == 'MEDIUM':
        report_content += "🟡 **中等可靠性** - PATE置信度适中(>0.7)，需要更多证据支撑"
    else:
        report_content += "🔴 **低可靠性** - 证据不足或存在冲突，建议进一步分析验证"

    report_content += f"""

## 📈 信号段投票统计

**故障类型投票分布**:
"""

    # 添加投票分布
    for fault_type, vote_count in file_result['fault_vote_distribution'].items():
        percentage = vote_count / file_result['segment_count'] * 100
        report_content += f"- {fault_type} ({fault_mapping.get(fault_type, fault_type)}): {vote_count}票 ({percentage:.1f}%)\n"

    report_content += f"""
**投票一致性分析**:
- **最高票数**: {max(file_result['fault_vote_distribution'].values())}票
- **投票率**: {file_result['vote_ratio']:.3f}
- **平均PATE置信度**: {file_result['average_pate_confidence']:.3f}
- **平均系统可靠性**: {file_result['average_reliability']:.3f}

## 🎯 诊断结论与建议

**最终诊断结论**:
根据闭环验证系统的三阶段分析，该轴承信号被诊断为**{final_diagnosis} ({fault_mapping.get(final_diagnosis, final_diagnosis)})**，系统置信度为**{confidence_level}**。

**证据支撑强度**:
"""

    # 根据不同情况给出结论建议
    if file_result['vote_ratio'] >= 0.8:
        report_content += "🟢 **强证据支撑** - 多段信号高度一致，诊断结果可靠\n"
    elif file_result['vote_ratio'] >= 0.6:
        report_content += "🟡 **中等证据支撑** - 多数段信号支持诊断，但存在一定分歧\n"
    else:
        report_content += "🔴 **弱证据支撑** - 段间诊断分歧较大，结果不确定性较高\n"

    if confidence_level == '高置信度':
        report_content += "\n**维护建议**: 立即关注，安排专项检查和维护计划"
    elif confidence_level == '中等置信度':
        report_content += "\n**维护建议**: 持续监控，建议在下次计划维护时重点检查"
    else:
        report_content += "\n**维护建议**: 需要进一步分析确认，可结合其他诊断手段验证"

    report_content += f"""

## 🔬 技术创新亮点

本诊断报告展现了闭环验证系统的核心创新：

1. **三阶段验证架构**: 假设生成 → 机理验证 → 专家推理的完整闭环
2. **异构证据融合**: 深度学习特征与物理机理证据的智能整合
3. **LLM逻辑推理**: 大语言模型作为专家会诊的核心推理引擎
4. **完整可解释性**: 每个诊断决策都有详细的证据链和推理过程
5. **质量可量化**: 多维度的诊断质量评估和可靠性分级

这种革命性的诊断框架实现了从"黑箱AI预测"到"透明专家会诊"的技术跃迁，为智能故障诊断领域开辟了全新路径。

---
*Generated by Revolutionary Closed-Loop Verification System*
*Mathematical Contest in Modeling 2025 - Problem E*
*Report Generation Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""

    # 保存报告
    report_filename = f"closed_loop_diagnostic_report_{file_name.replace('.mat', '')}.md"
    report_path = os.path.join(output_dir, report_filename)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    return report_path

def main():
    """主函数：为所有16个目标域文件生成个人报告"""
    print("=== 闭环验证系统 - 个人诊断报告生成 ===")

    # 加载结果
    results = load_closed_loop_results()

    print(f"发现 {len(results)} 个诊断结果，开始生成个人报告...")

    report_paths = []

    for i, file_result in enumerate(results):
        file_name = file_result['file_name']
        print(f"  生成报告 {i+1}/{len(results)}: {file_name}")

        try:
            report_path = generate_individual_report(file_result)
            report_paths.append(report_path)
            print(f"    ✓ 报告已保存: {report_path}")
        except Exception as e:
            print(f"    ✗ 生成失败: {e}")

    print(f"\n✅ 成功生成 {len(report_paths)} 个个人诊断报告")
    print(f"报告保存位置: results/individual_reports/")

    # 生成总结索引
    generate_index_file(results, report_paths)

def generate_index_file(results, report_paths):
    """生成报告索引文件"""
    index_content = f"""# 闭环验证系统 - 个人诊断报告索引

## 📋 报告概览
- **生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **总报告数**: {len(report_paths)}个
- **诊断文件**: A.mat - P.mat (目标域列车轴承数据)

## 📊 诊断结果汇总

| 文件 | 最终诊断 | 置信度 | 投票率 | PATE置信度 | 系统可靠性 | 报告链接 |
|------|----------|--------|--------|------------|------------|----------|
"""

    for result in results:
        file_name = result['file_name'].replace('.mat', '')
        final_diagnosis = result['final_diagnosis']
        confidence = result['confidence_level']
        vote_ratio = result['vote_ratio']
        pate_conf = result['average_pate_confidence']
        reliability = result['average_reliability']

        report_link = f"[详细报告](closed_loop_diagnostic_report_{file_name}.md)"

        index_content += f"| {result['file_name']} | {final_diagnosis} | {confidence} | {vote_ratio:.3f} | {pate_conf:.3f} | {reliability:.3f} | {report_link} |\n"

    # 添加统计信息
    fault_dist = {}
    for result in results:
        fault = result['final_diagnosis']
        fault_dist[fault] = fault_dist.get(fault, 0) + 1

    index_content += f"""
## 📈 统计分析

### 故障分布
"""
    for fault, count in fault_dist.items():
        percentage = count / len(results) * 100
        index_content += f"- **{fault}**: {count}个文件 ({percentage:.1f}%)\n"

    avg_vote_ratio = sum(r['vote_ratio'] for r in results) / len(results)
    avg_pate_conf = sum(r['average_pate_confidence'] for r in results) / len(results)
    avg_reliability = sum(r['average_reliability'] for r in results) / len(results)

    index_content += f"""
### 平均性能指标
- **平均投票一致率**: {avg_vote_ratio:.3f}
- **平均PATE置信度**: {avg_pate_conf:.3f}
- **平均系统可靠性**: {avg_reliability:.3f}

## 🔗 快速导航

### 按诊断结果分类
"""

    for fault in ['Inner', 'Outer', 'Ball', 'Normal']:
        if fault in fault_dist:
            index_content += f"\n**{fault}故障 ({fault_dist[fault]}个)**:\n"
            for result in results:
                if result['final_diagnosis'] == fault:
                    file_name = result['file_name'].replace('.mat', '')
                    confidence = result['confidence_level']
                    index_content += f"- [{result['file_name']}](closed_loop_diagnostic_report_{file_name}.md) ({confidence})\n"

    index_content += """
---
*Generated by Revolutionary Closed-Loop Verification System*
*Mathematical Contest in Modeling 2025 - Problem E*
"""

    # 保存索引文件
    index_path = os.path.join('results/individual_reports', 'README.md')
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(index_content)

    print(f"✅ 报告索引已生成: {index_path}")

if __name__ == '__main__':
    main()