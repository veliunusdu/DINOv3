import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import { Workbook } from 'file:///C:/Users/veliu/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';

// Read the final report, not an older benchmark session.
const root='C:/Codes/Projects/DINOv3';
const report=await fs.readFile(`${root}/Docs/DINOv3_Final_Deney_Raporu_TR.md`,'utf8');
const quality=report.split('## Ana doğruluk değerlendirmesi — 1.399 görüntü')[1].split('## Son süre benchmark')[0];
const timing=report.split('## Son süre benchmark')[1].split('## ')[0];
function rows(section){return section.split('\n').filter(l=>l.startsWith('|')).map(l=>l.split('|').slice(1,-1).map(v=>v.trim()));}
const q=new Map(rows(quality).filter(r=>r.length===3).map(r=>[r[0],r]));
const t=new Map(rows(timing).filter(r=>r.length===7).map(r=>[r[0],r]));
const num=s=>Number(s.replace(',','.'));
const specs=[['Baseline',0,'n.a.'],['Sliding',25,112],['Random 42 K=3',3,112],['Random 43 K=3',3,112],['Random 44 K=3',3,112],['Random K=3 three-seed mean',3,112],['Adaptive K=1',1,112],['Adaptive K=3',3,112]];
const seeds=['Random 42 K=3','Random 43 K=3','Random 44 K=3'];
const mean=(a)=>a.reduce((x,y)=>x+y,0)/a.length;
const baseIoU=num(q.get('Baseline')[1]),baseMs=num(t.get('Baseline')[2]);
const header=['Method','K','Crop Size','mIoU','Delta mIoU','Inference Time','Extra Inference','Relative Cost'];
const matrix=[header];
for(const [method,k,crop] of specs){
  const summary=method==='Random K=3 three-seed mean';
  const miou=summary?mean(seeds.map(s=>num(q.get(s)[1]))):num(q.get(method)[1]);
  const ms=summary?mean(seeds.map(s=>num(t.get(s)[2]))):num(t.get(method)[2]);
  const passes=summary?4:Number(t.get(method)[6]);
  assert(Number.isFinite(miou)&&Number.isFinite(ms)&&Number.isInteger(passes));
  if(!summary) assert.equal(passes,Number(q.get(method)[2]));
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
