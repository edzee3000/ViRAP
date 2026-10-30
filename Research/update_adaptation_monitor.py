"""Refresh monitor assets and the preview sidecar, leaving trainers untouched."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
path=root/'Model/mae_training_dashboard.html'
page=path.read_text(encoding='utf-8')
page=page.replace("$('parallelCount').textContent=expected+' GPU';", "$('parallelCount').textContent=running+' GPU / '+expected+'组实验';")
page=page.replace("s.stage==='hybrid'?'前8层LoRA＋末4层全量':s.stage==='joint'?'末4层联合微调':'冻结骨干训练'",
                  "s.stage==='all_lora'?'冻结MAE＋全12层LoRA':s.stage==='full'?'MAE全参数微调':s.stage==='hybrid'?'前8层LoRA＋末4层全量':s.stage==='joint'?'末4层联合微调':'冻结骨干训练'")
path.write_text(page,encoding='utf-8')
print('adaptation_monitor_labels_ready')
