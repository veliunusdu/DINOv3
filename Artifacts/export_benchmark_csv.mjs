import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import { Workbook } from 'file:///C:/Users/veliu/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';

// Use verified full-precision results; round only the exported display values.
const root='C:/Codes/Projects/DINOv3';
const quality=JSON.parse(await fs.readFile(`${root}/Reproducibility/metrics_verification.json`,'utf8'));
assert.equal(quality.status,'PASS');
const timing=JSON.parse(await fs.readFile(`${root}/Deliverables/Quality_Cost/figure_data.json`,'utf8')).timing_100;
const keys={'Baseline':'baseline','Sliding':'sliding','Adaptive K=1':'adaptive_k1','Adaptive K=3':'adaptive_k3',
  'Random K=3 three-seed mean':'random_k3_mean',...Object.fromEntries([42,43,44].map(s=>[`Random ${s} K=3`,`random_${s}_k3`]))};
const specs=[['Baseline',0,'n.a.'],['Sliding',25,112],['Random 42 K=3',3,112],['Random 43 K=3',3,112],['Random 44 K=3',3,112],['Random K=3 three-seed mean',3,112],['Adaptive K=1',1,112],['Adaptive K=3',3,112]];
const seeds=['Random 42 K=3','Random 43 K=3','Random 44 K=3'];
const mean=(a)=>a.reduce((x,y)=>x+y,0)/a.length;
const baseIoU=quality.pooled_miou.baseline,baseMs=timing.baseline.mean_ms;
const header=['Method','K','Crop Size','mIoU','Delta mIoU','Inference Time','Extra Inference','Relative Cost'];
const matrix=[header];
for(const [method,k,crop] of specs){
  const summary=method==='Random K=3 three-seed mean';
  const miou=quality.pooled_miou[keys[method]];
  const ms=summary?mean(seeds.map(s=>timing[keys[s]].mean_ms)):timing[keys[method]].mean_ms;
  const passes=timing[keys[method]].passes;
  assert(Number.isFinite(miou)&&Number.isFinite(ms)&&Number.isInteger(passes));
  assert.equal(passes,method==='Baseline'?1:method==='Sliding'?25:k+1);
  matrix.push([method==='Sliding'?'Sliding-window':method,k,crop,miou.toFixed(4),((miou-baseIoU)*100).toFixed(2),ms.toFixed(1),passes-1,(ms/baseMs).toFixed(4)]);
}
assert.equal(matrix.length,9);assert.equal(matrix[1][6],0);
const wb=Workbook.create();const sheet=wb.worksheets.add('Comparison');
sheet.getRange('A1:H9').values=matrix;
await wb.recalculate();
const stored=sheet.getRange('A1:H9').values;
assert.deepEqual(stored,matrix);
const quote=v=>/[",\r\n]/.test(String(v))?'"'+String(v).replaceAll('"','""')+'"':String(v);
const csv=stored.map(r=>r.map(quote).join(',')).join('\r\n')+'\r\n';
const dest=`${root}/Deliverables/DINOv3_Method_Comparison.csv`;
await fs.writeFile(dest,'\uFEFF'+csv,'utf8');
// Round-trip CSV parsing verifies delimiters, row count and all saved values.
const saved=await fs.readFile(dest,'utf8');
assert.equal(saved.charCodeAt(0),0xFEFF);
const check=await Workbook.fromCSV(saved.slice(1),{sheetName:'Check'});
assert.deepEqual(check.worksheets.getItem('Check').getRange('A1:H9').values,matrix.map(r=>r.map(String)));
console.log(csv);
console.log('CSV round-trip passed:',dest);
