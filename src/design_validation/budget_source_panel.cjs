/* Deterministic installed-source calls; no browser, tenant or human evidence. */
'use strict';
const E=require('./finite_schedule_engine.cjs');
function sourcePanel(design,seed){
  E.validateDesign(design);
  const schedules=[], templates=new Map(),beliefCases=[],outcomeCases=[];
  for(const conditionId of ['C1','C2','C3','C4','C5'])for(let variant=0;variant<9;variant++){
    const id='budget-source-'+conditionId+'-v'+variant;
    const options={conditionId,scheduleVariant:variant,seed:String(seed),participantId:id,sessionId:id+'-session',profileId:E.ENGAGEMENT};
    const definitions=E.makeSchedule(design,options).filter(d=>d.stage==='main').map(d=>({...d,capacity:E.questionCapacity(options,d)}));
    schedules.push({conditionId,variant,definitions});
    definitions.forEach(d=>templates.set(d.templateId,d));
  }
  for(const [templateId,d]of templates){
    for(let student=0;student<2;student++){
      let logs=E.prior(d.priorCentres[student],d.priorSpreads[student]);
      for(const reports of [[],[1],[1,50],[1,50,100]]){
        logs=E.prior(d.priorCentres[student],d.priorSpreads[student]);
        for(const y of reports)logs=E.posterior(logs,y,E.likelihood(d.reportWidth));
        const m=E.moments(logs);
        beliefCases.push({templateId,student,reports,probabilities:m.probabilities,mean:m.mean,sd:m.sd,predictiveRMSE:E.predictiveRMSE(logs,E.likelihood(d.reportWidth))});
      }
    }
    for(const budget of [d.budgetMillihours,d.budgetMillihours-d.costMillihours,0])
      for(const needs of [[1,100],[50,70],[45,65],[100,1]])
        for(const index of budget?[0,1,499,500,501,999,1000]:[null]){
          const r=E.outcome(budget,index,needs);
          outcomeCases.push({templateId,budget,index,needs,amountsMicrohours:r.amountsMicrohours,utility:r.totalUtility,gpa:r.totalGPA});
        }
  }
  return {schema:'budget-finite-source-panel-v1',profileId:E.ENGAGEMENT,seed,schedules,beliefCases,outcomeCases};
}
module.exports={sourcePanel};
if(require.main===module){let raw='';process.stdin.setEncoding('utf8');process.stdin.on('data',s=>raw+=s);process.stdin.on('end',()=>{
  try{const r=JSON.parse(raw);process.stdout.write(JSON.stringify(sourcePanel(r.design,r.seed)));}
  catch(e){process.stderr.write(String(e.message));process.exitCode=1;}
});}
