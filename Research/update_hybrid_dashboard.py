"""Update labels for the additional hybrid experiment; no training changes."""
from pathlib import Path

p=Path(__file__).resolve().parents[1]/'Model/mae_training_dashboard.html'
text=p.read_text(encoding='utf-8')
if 'id="parallelCount"' not in text:
    text=text.replace('8 GPU并行','<span id="parallelCount">8 GPU</span>并行')
    text=text.replace("const expected=state.supervisor.jobs?.length||8;", "const expected=state.supervisor.jobs?.length||8;$('parallelCount').textContent=expected+' GPU';")
    text=text.replace("s.stage==='joint'?'末4层联合微调':'冻结骨干训练'", "s.stage==='hybrid'?'前8层LoRA＋末4层全量':s.stage==='joint'?'末4层联合微调':'冻结骨干训练'")
    text=text.replace('${run.config.accumulate}', '${run.config.effective_batch??run.config.accumulate}')
    text=text.replace('K=${s.local_rank} · seed=${s.seed}', 'K=${s.local_rank} · seed=${s.seed}${s.lora_rank?" · LoRA r="+s.lora_rank:""}')
    p.write_text(text,encoding='utf-8')
print('hybrid_dashboard_labels_ready')
