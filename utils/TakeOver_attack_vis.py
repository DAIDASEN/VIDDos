import matplotlib.pyplot as plt
import numpy as np


attack_start_t = 5  
timestamps = list(range(1, 21))
latency_raw = [1.0, 1.0, 1.0, 1.0, 1.0, 
               5.5, 4.0, 5.5, 4.5, 5.0, 4.5, 3.5, 5.8, 5.5, 5.5, 5.8, 4.5, 6.8, 5.0, 6.0]
tau_safe = 2.72 # 安全阈值


# 计算累积延迟
cumulative_latency = []
current_cumulative = 0
for i, l in enumerate(latency_raw):
    if i > 0 and cumulative_latency[-1] > tau_safe:
        # 如果上一个时刻超时，延迟累积到当前
        carry_over = cumulative_latency[-1] - tau_safe
        current_cumulative = l + carry_over
    else:
        current_cumulative = l
    cumulative_latency.append(current_cumulative)


fig, ax = plt.subplots(figsize=(10, 6))

y_max = max(cumulative_latency) + 1.5
ax.axhspan(0, tau_safe, facecolor='green', alpha=0.1)      # 安全区：绿色
ax.axhspan(tau_safe, y_max, facecolor='red', alpha=0.1)    # 违规区：红色

bars = ax.bar(timestamps, latency_raw, width=0.6, alpha=0.7, 
              color='skyblue', edgecolor='blue', linewidth=1, label='Inherent Latency (s)')

ax.plot(timestamps, cumulative_latency, marker='o', linestyle='-', 
        color='red', linewidth=2, markersize=6, label='Cumulative Latency (s)')

# 标记攻击开始
ax.axvline(x=attack_start_t + 1 - 0.5, color='orange', linestyle='--', linewidth=2, alpha=0.8, label='Attack Start (t=5)')

ax.axhline(y=tau_safe, color='green', linestyle='--', linewidth=2, label=f'Safety Threshold ($\\tau_{{safe}}$ = {tau_safe} s)')

for i, (t, raw, cum) in enumerate(zip(timestamps, latency_raw, cumulative_latency)):
    if cum > tau_safe:
        ax.text(t, cum + 0.2, f'{cum:.1f}', ha='center', fontsize=9, color='darkred', fontweight='bold')
    else:
        ax.text(t, raw + 0.2, f'{raw:.1f}', ha='center', fontsize=9, color='blue')

ax.set_xlabel('Decision Timestamp $t$', fontsize=12)
ax.set_ylabel('Latency (s)', fontsize=12)
ax.set_title('Real-time Takeover Latency under VidDoS Attack\n(with Cumulative Delay Effect)', fontsize=14)
ax.set_xticks(timestamps)
ax.set_ylim(0, max(cumulative_latency) + 1.5)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.show()