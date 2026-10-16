#!/usr/bin/env python3
"""Historical stochastic calibration with deterministic immutable checkpoints."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from tutoring_analysis import calibration as C
from tutoring_analysis.reproduction import write_historical_failures
import numpy as np


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(data):return (json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()


def write_new(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as handle:handle.write(encoded(data));handle.flush();os.fsync(handle.fileno())


def replace_ledger(path,data):
    temporary=Path(str(path)+".next")
    write_new(temporary,data);os.replace(temporary,path)


def source_hashes():
    names=["src/tutoring_analysis/calibration.py","src/tutoring_analysis/inference.py","src/tutoring_analysis/planning.py",
           "src/tutoring_analysis/reproduction.py",
           "scripts/calibrate_tutoring_methods.py","tests/test_tutoring_calibration.py"]
    return {name:digest(ROOT/name) for name in names if (ROOT/name).exists()}


def validate_ledger(ledger,output,cells,targets,batch_size):
    expected=[(c["id"],i) for c in cells for i in range(targets[c["id"]]//batch_size)]
    entries=ledger["batches"]
    if [(e["cell"],e["batchIndex"]) for e in entries]!=expected[:len(entries)]:raise ValueError("Checkpoint is not unique ordered target prefix")
    by_id={c["id"]:c for c in cells}
    for e in entries:
        relative=f"cells/{e['cell']}/batch_{e['batchIndex']:04d}.json"
        if e["file"]!=relative or e["attempts"]!=batch_size:raise ValueError("Checkpoint file/count identity mismatch")
        if digest(output/relative)!=e["sha256"]:raise ValueError("Checkpoint hash mismatch")
        data=json.loads((output/relative).read_text())
        if (data["cell"],data["batchIndex"],data["attempts"],data["seed"],data["firstReplica"])!=(e["cell"],e["batchIndex"],batch_size,by_id[e["cell"]]["seed"],e["batchIndex"]*batch_size):raise ValueError("Batch payload identity mismatch")


def csv_new(path,rows):
    if not rows:return
    with Path(path).open("x",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def readable_report(summaries,config,mode):
    lines=["# Finite synthetic method calibration","",
        "This is empirical synthetic evidence about the declared statistical generators, not human data, behavioral adaptation, finite-task policy validation, approval, or a final sample-size choice.","",
        f"Execution mode: {mode}. Fixed global nine-slot Holm is the historical primary policy. Supporting routes and the two alternative policies cannot rescue a primary result.","",
        "All rejection frequencies use attempted replications. Coverage uses available pointwise intervals with missing counts shown. Binomial intervals are two-sided 95% exact Clopper–Pearson. Normal-summary cells are an abstract idealization; bounded, skew and selection stress have their own limited scope.","",
        "| Cell | Attempts | Prospective classification | Primary coverage failures | Global true-null FWER upper CI | Unexpected supported failures |",
        "| --- | ---: | --- | ---: | ---: | ---: |"]
    for s in summaries:
        p=s["populations"]["primary"];f=p["policies"]["global_holm"]["trueNullFWER"]
        fails=sum(t["coverageCriterionPassed"] is False for t in p["tests"])
        lines.append(f"| {s['cell']['id']} | {p['attempts']} | {s['cell']['nominalStatus']} | {fails} | {f['interval'][1]:.5f} | {p['unexpectedSupportedFailures']} |")
    lines.extend(["","No failed cell was replaced or relabelled. Inspect calibration_summary.json and each cell's summary.json for every window, support count, exact target, three policies, marginal error, mixed-null discoveries, selected-target account, low-tail criterion and power comparison.","",
        "MNAR original-population coverage/bias and nominal-null rejection diagnose selection. Nonzero selected targets are alternatives, not selected-target type-I errors. Complete-ten means ten records; endpoint observations may be fewer. Early/late windows are stages 1–5/6–10; eight-record histories are sensitivities and dose 30 is hypothetical.","",
        "Full-row 84-route checks are arithmetic parity under a surrogate paired-target geometry, not stochastic calibration or genuine finite exports. Gaussian sampled replicas are compared separately against the unchanged production contrast, person OLS and multiplicity APIs.","",
        "Conditional power comparisons use known generator moments, never fitted QA moments. Heterogeneous selected moments/counts without an established common planning distribution are marked not comparable. Joint Holm bounds are conservative conditional marginal bounds, not an exact joint-power model.","",
        "Synthetic calibration is not evidence of human behavior or a final sample-size decision.",""])
    return "\n".join(lines)


def finalize(output,config,cells,ledger,mode):
    summaries=[];core_rows=[];error_rows=[];power_rows=[]
    for cell in cells:
        records=[json.loads((output/e["file"]).read_text()) for e in ledger["batches"] if e["cell"]==cell["id"]]
        s=C.summarize_cell(cell,config,records);summaries.append(s)
        write_new(output/"cells"/cell["id"]/"summary.json",s)
        for population,p in s["populations"].items():
            for t in p["tests"]:
                cov=t["coverage"];sel=t["selectedCoverage"];err=t["rawFalseReject"]
                core_rows.append({"cell":cell["id"],"classification":cell["nominalStatus"],"population":population,"id":t["id"],
                    "attempts":t["attempts"],"available":t["available"],"unavailable":t["unavailable"],"meanEstimate":t["meanEstimate"],
                    "meanPopulationTruth":t["meanTruth"],"meanSelectedTruth":t["meanSelectedTruth"],"coverage":cov["rate"],
                    "coverageDenominator":cov["denominator"],"coverageLow":cov["interval"][0] if cov["interval"] else None,
                    "coverageHigh":cov["interval"][1] if cov["interval"] else None,"selectedCoverage":sel["rate"],
                    "rawFalseReject":err["rate"],"rawFalseRejectUpper":err["interval"][1],
                    "coveragePassed":t["coverageCriterionPassed"],"rawErrorPassed":t["rawErrorCriterionPassed"],
                    "lowTailPassed":t["designatedLowTailCriterionPassed"],"unexpectedNumericalFailure":t["unexpectedNumericalFailure"]})
            for policy,r in p["policies"].items():
                f=r["trueNullFWER"]
                error_rows.append({"cell":cell["id"],"classification":cell["nominalStatus"],"population":population,"policy":policy,
                    "attempts":p["attempts"],"trueNullFWER":f["rate"],"FWERLower":f["interval"][0],"FWERUpper":f["interval"][1],
                    "globalCriterionPassed":r["globalErrorCriterionPassed"],"globalClaim":r["globalGuarantee"],
                    "anyCoreReject":r["anyCoreReject"]["rate"],"allCoreReject":r["allCoreReject"]["rate"],
                    "alternativeSubsetAttempts":r["alternativeSubsetAttempts"],"alternativePowerStatus":r["alternativeDiscoveryPowerStatus"],
                    "anyAlternativeDiscovery":r["anyAlternativeDiscovery"]["rate"] if r["anyAlternativeDiscovery"] else None,
                    "allAlternativeDiscovery":r["allAlternativeDiscovery"]["rate"] if r["allAlternativeDiscovery"] else None})
        for r in s["powerComparison"]["comparisons"]:
            power_rows.append({"cell":cell["id"],"id":r["id"],"threshold":r["threshold"],"planningStatus":r["planning"]["status"],
                "planningPower":r["planning"]["power"],"empirical":r["empirical"]["rate"],"empiricalLower":r["empirical"]["interval"][0],
                "empiricalUpper":r["empirical"]["interval"][1],"absoluteDifference":r["absoluteDifference"],"tolerancePassed":r["withinFrozenDifferenceTolerance"]})
    write_new(output/"calibration_summary.json",{"mode":mode,"purpose":config["purpose"],"attempts":sum(s["populations"]["primary"]["attempts"] for s in summaries),
              "cellCount":len(summaries),"cells":summaries,"criteria":config["criteria"],"noHumanInference":True})
    csv_new(output/"core_results.csv",core_rows);csv_new(output/"policy_error_results.csv",error_rows);csv_new(output/"calibration_vs_power.csv",power_rows)
    write_historical_failures(output, {"cells": summaries, "criteria": config["criteria"],
        "attempts": sum(s["populations"]["primary"]["attempts"] for s in summaries)})
    with (output/"methods_and_limits.md").open("x") as handle:handle.write(readable_report(summaries,config,mode))
    hashes={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name not in ("completion.json","checkpoint.json")}
    write_new(output/"completion.json",{"status":"completed_frozen_attempts_not_all_criteria_pass","mode":mode,"binding":ledger["binding"],
              "attempts":sum(s["populations"]["primary"]["attempts"] for s in summaries),"batchCount":len(ledger["batches"]),
              "outputHashes":hashes,"sourceHashes":source_hashes(),"elapsedBatchSeconds":sum(e["elapsedSeconds"] for e in ledger["batches"])})


def run(args):
    config_path=Path(args.config).resolve();config=json.loads(config_path.read_text());cells=C.resolved_cells(config)
    if config["batchSize"]<1 or any(c["replications"]%config["batchSize"] for c in cells):raise ValueError("Fixed full counts must divide fixed batch size")
    output=Path(args.output).resolve();source=source_hashes()
    binding={"configHash":digest(config_path),"sourceHashes":source,"mode":args.mode,"schema":"strict_calibration_checkpoint_v1",
             "python":platform.python_version(),"numpy":np.__version__,"scipy":__import__("scipy").__version__}
    targets={c["id"]:c["replications"] if args.mode=="full" else config["batchSize"] for c in cells}
    ledger_path=output/"checkpoint.json"
    if args.resume:
        ledger=json.loads(ledger_path.read_text())
        if ledger["binding"]!=binding:raise ValueError("Resume source/config/runtime/mode binding mismatch")
        validate_ledger(ledger,output,cells,targets,config["batchSize"])
        if (output/"completion.json").exists():
            receipt=json.loads((output/"completion.json").read_text())
            for name,sha in receipt["outputHashes"].items():
                if digest(output/name)!=sha:raise ValueError("Completed output hash mismatch")
            print(json.dumps({"status":"verified_existing_completion","output":str(output),"attempts":receipt["attempts"]}));return
        if ledger["batches"]:
            last=ledger["batches"][-1];cell=next(c for c in cells if c["id"]==last["cell"])
            replay=C.evaluate_batch(cell,config,last["batchIndex"],last["attempts"])
            if encoded(replay)!=(output/last["file"]).read_bytes():raise ValueError("Last checkpoint deterministic replay mismatch")
    else:
        if output.exists() and any(output.iterdir()):raise ValueError("Refusing to overwrite nonempty output; use exact --resume or fresh directory")
        output.mkdir(parents=True,exist_ok=True);ledger={"binding":binding,"batches":[]}
        write_new(ledger_path,ledger)
        write_new(output/"cell_manifest.json",{"config":config,"resolvedCells":cells,"targets":targets,"binding":binding,
                  "argv":sys.argv,"purpose":config["purpose"],"classificationFrozenBeforeExecution":True})
        write_new(output/"full_row_parity.json",C.full_row_parity(config))
    seen={(e["cell"],e["batchIndex"]) for e in ledger["batches"]}
    count_this_run=0
    for cell in cells:
        for index in range(targets[cell["id"]]//config["batchSize"]):
            if (cell["id"],index) in seen:continue
            started=time.perf_counter()
            try:
                data=C.evaluate_batch(cell,config,index,config["batchSize"])
                relative=f"cells/{cell['id']}/batch_{index:04d}.json";write_new(output/relative,data)
                ledger["batches"].append({"cell":cell["id"],"batchIndex":index,"attempts":config["batchSize"],"file":relative,
                       "sha256":digest(output/relative),"elapsedSeconds":time.perf_counter()-started})
                replace_ledger(ledger_path,ledger)
            except Exception as exc:
                write_new(output/f"failure_{cell['id']}_{index:04d}_{time.time_ns()}.json",{"cell":cell["id"],"batchIndex":index,
                          "binding":binding,"errorType":type(exc).__name__,"message":str(exc)})
                raise
            count_this_run+=1
            if index%10==0:print(json.dumps({"cell":cell["id"],"batch":index,"seconds":ledger["batches"][-1]["elapsedSeconds"]}),flush=True)
            if args.stop_after_batches and count_this_run>=args.stop_after_batches:
                print(json.dumps({"status":"intentional_checkpoint_stop","completedBatches":len(ledger["batches"])}));return
    finalize(output,config,cells,ledger,args.mode)
    print(json.dumps({"status":"completed","output":str(output),"attempts":sum(targets.values())}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default=str(ROOT/"configs/tutoring_calibration_v01.json"))
    parser.add_argument("--output",required=True)
    parser.add_argument("--mode",choices=("smoke","full"),default="smoke")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--stop-after-batches",type=int,help="Explicit interruption test; preserves incomplete checkpoint, never claims calibration completion")
    args=parser.parse_args()
    if args.stop_after_batches is not None and args.stop_after_batches<1:parser.error("positive stop count required")
    run(args)


if __name__=="__main__":main()
