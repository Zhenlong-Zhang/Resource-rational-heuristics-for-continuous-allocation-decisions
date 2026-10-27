"""Portable, source-bound execution helpers for read-only saved-model checks."""
import shutil
import os
import subprocess
import sys
from pathlib import Path

from .saved_fit_inputs import load, write_new
from .saved_fit_magnitude import require

ROOT = Path(__file__).resolve().parent
SOURCE_NAMES = ("saved_fit_magnitude.py", "saved_fit_inputs.py", "saved_fit_runtime.py",
                "saved_fit_reproduction.py", "independent_saved_fit.py",
                "independent_saved_fit.R", "magnitude_composition.py",
                "independent_cluster.py", "independent_cluster_rds.R", "source_transport.R",
                "power_reference.py", "__init__.py", "data/numerical_environment.json")


def sources(observed):
    paths = {name: ROOT / name for name in SOURCE_NAMES}
    paths["sourcePackage"] = ROOT.parent / "__init__.py"
    paths["publicEntry"] = ROOT.parents[1] / "scripts/reproduce_saved_fit_magnitude.py"
    return {name: observed.actual(path) for name, path in sorted(paths.items())}


PROBE = r'''
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
expected <- read_json(args[[1L]],simplifyVector=FALSE)
stopifnot(identical(as.character(getRversion()),expected$R))
packages <- list()
for (name in names(expected$R_packages)) {
  stopifnot(requireNamespace(name,quietly=TRUE))
  description <- normalizePath(file.path(find.package(name),"DESCRIPTION"),mustWork=TRUE)
  version <- unname(read.dcf(description)[1L,"Version"])
  stopifnot(identical(version,expected$R_packages[[name]]))
  packages[[name]] <- list(description=description,version=version)
}
cat(toJSON(list(R=as.character(getRversion()),Rhome=normalizePath(R.home()),
  packages=packages),auto_unbox=TRUE,digits=NA,null="null"))
'''


def runtime(rscript, observed, output):
    import numpy
    import scipy
    expected = load(ROOT / "data/numerical_environment.json")["statistical_reference"]
    require(sys.version.split()[0] == expected["python"] and numpy.__version__ == expected["numpy"]
            and scipy.__version__ == expected["scipy"], "Use the declared reference numerical environment")
    executable = shutil.which(str(rscript))
    require(executable is not None, "Rscript is not available")
    executable = str(Path(executable).resolve())
    for path in (Path(sys.executable).resolve(), Path(executable)):
        require(observed.resolve(path) == path, "Cannot relocate executing runtime")
    pins = [observed.actual(Path(sys.executable).resolve()), observed.actual(executable)]
    rp, ro, log = (output.with_suffix(suffix) for suffix in
                   (".runtime_request.json", ".runtime.json", ".runtime.log"))
    write_new(rp, expected)
    with ro.open("x") as stream, log.open("x") as errors:
        subprocess.run([executable, "--vanilla", "-e", PROBE, str(rp)],
                       stdout=stream, stderr=errors, check=True)
    actual = load(ro)
    require(set(actual["packages"]) == set(expected["R_packages"]), "Incomplete runtime inventory")
    for name, item in actual["packages"].items():
        require(item["version"] == expected["R_packages"][name], "Changed selected R package")
        require(observed.resolve(item["description"]) == Path(item["description"]).resolve(),
                "Cannot relocate loaded runtime package")
        pins.append(observed.actual(item["description"]))
    for path in (rp, ro, log):
        observed.actual(path)
    return dict(python=sys.version.split()[0], numpy=numpy.__version__, scipy=scipy.__version__,
                Rscript=executable, packageVersions=expected["R_packages"], actualR=actual,
                runtimeFiles={pin["path"]: pin["sha256"] for pin in pins},
                identity="current_reference_execution_not_historical_binary_identity")


R_BOUND_ENTRY = r'''
suppressPackageStartupMessages(library(jsonlite))
request <- read_json(commandArgs(trailingOnly=TRUE)[[1L]],simplifyVector=FALSE)
binding <- request$runtimeBinding
stopifnot(identical(as.character(getRversion()),"4.2.2"))
hash <- Sys.which("sha256sum")
stopifnot(nzchar(hash))
for (name in names(binding$packageVersions)) {
  stopifnot(requireNamespace(name,quietly=TRUE))
  path <- normalizePath(file.path(find.package(name),"DESCRIPTION"),mustWork=TRUE)
  stopifnot(identical(unname(read.dcf(path)[1L,"Version"]),binding$packageVersions[[name]]))
  line <- system2(hash,shQuote(path),stdout=TRUE,stderr=TRUE)
  stopifnot(is.null(attr(line,"status")),length(line)==1L,
            identical(substr(line,1L,64L),binding$runtimeFiles[[path]]))
}
source(Sys.getenv("SAVED_FIT_R_HELPER"),chdir=FALSE)
'''


def run_r(rscript, script, request, output, observed, runtime_binding=None):
    rp, log = output.with_suffix(".request.json"), output.with_suffix(".log")
    require(not output.exists(), "Existing R output; choose a new destination")
    if runtime_binding is not None:
        request = dict(request, runtimeBinding={key: runtime_binding[key] for key in ("packageVersions", "runtimeFiles")})
    write_new(rp, request)
    args = [rscript, "--vanilla", str(script), str(rp), str(output)]
    env = None
    if runtime_binding is not None:
        args = [rscript, "--vanilla", "-e", R_BOUND_ENTRY, str(rp), str(output)]
        env = dict(os.environ, SAVED_FIT_R_HELPER=str(script))
    with log.open("x") as stream:
        subprocess.run(args, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    artifacts = {name: observed.actual(path) for name, path in
                 (("request", rp), ("output", output), ("log", log))}
    return load(output), artifacts
