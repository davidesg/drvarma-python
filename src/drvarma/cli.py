"""Command-line interface mirroring the drvarma C binary.

    drvarma <file> <p> <q> [options]

`<file>` is the base name: input is read from ``<file>.inp`` and results are
written to ``<file>.out`` (always), plus ``<file>.forecast`` (with -forecast)
and ``<file>.recursive`` (with -estwin).  The Box-Cox lambda and differencing
orders are taken from the ``.inp`` header, as in the C engine.
"""

import argparse
import os
import sys

from . import load, Model, report


def build_parser():
    p = argparse.ArgumentParser(
        prog="drvarma",
        description="Multivariate VARMA estimation, forecasting and diagnostics.")
    p.add_argument("file", help="base name (input read from <file>.inp)")
    p.add_argument("p", type=int, help="AR order")
    p.add_argument("q", type=int, help="MA order")
    p.add_argument("-mean", action="store_true", help="estimate a mean/drift term")
    p.add_argument("-diagar", action="store_true", help="diagonal AR matrices")
    p.add_argument("-diagma", action="store_true", help="diagonal MA matrices")
    p.add_argument("-diagcov", action="store_true", help="diagonal covariance")
    p.add_argument("-m", type=int, default=1, dest="method",
                   help="estimation method: 1=exact (default), 2=approximate")
    p.add_argument("-twostep", action="store_true",
                   help="two-step (Hannan-Rissanen) initialisation (q>0)")
    p.add_argument("-deseason", nargs="?", const="auto", choices=["auto", "force"],
                   default=None, help="harmonic seasonal adjustment (auto|force)")
    p.add_argument("-scale", type=float, default=100.0,
                   help="rescale factor after Box-Cox (default 100)")
    p.add_argument("-forecast", type=int, default=None, metavar="H",
                   help="forecast H steps; writes <file>.forecast")
    p.add_argument("-html", action="store_true",
                   help="also write an HTML SPS forecast report per series "
                        "(<file>_<series>.html; needs -forecast and jinja2)")
    p.add_argument("-estwin", type=int, default=None, metavar="N",
                   help="fixed-parameter recursive forecasts on first N raw obs; "
                        "writes <file>.recursive (needs -forecast)")
    p.add_argument("-volexp", nargs="*", default=None, metavar="ARG",
                   help="exponential volatility [alpha window] (defaults 0.05 20); "
                        "writes <file>.volexp")
    p.add_argument("-volmov", nargs="*", default=None, metavar="window",
                   help="moving-window volatility [window] (default 20); "
                        "writes <file>.volmov")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    base = args.file[:-4] if args.file.endswith(".inp") else args.file
    inp_path = base + ".inp"
    if not os.path.exists(inp_path):
        sys.exit("ERROR: cannot open %s" % inp_path)

    series, spec = load(inp_path)
    model = Model(series, lam=spec.lam, d=spec.d, D=spec.D, scale=args.scale,
                  p=args.p, q=args.q, include_mean=args.mean,
                  diag_ar=args.diagar, diag_ma=args.diagma, diag_cov=args.diagcov,
                  method=args.method, twostep=args.twostep, deseason=args.deseason)
    model.fit()

    if model.ifault != 0:
        print("Estimation error (ifault=%d)" % model.ifault, file=sys.stderr)

    out_path = base + ".out"
    report.write_out(model, out_path, input_path=inp_path, output_path=out_path)
    print("Results written to %s" % out_path)

    if args.forecast:
        fc_path = base + ".forecast"
        report.write_forecast(model, args.forecast, fc_path)
        print("Forecasts written to %s" % fc_path)
        if args.html:
            from . import report_forecast
            paths = report_forecast.write_forecast_report(model, base,
                                                          L=args.forecast)
            print("HTML forecast reports written to %s" % ", ".join(paths))
    elif args.html:
        sys.exit("ERROR: -html requires -forecast H")

    if args.estwin:
        if args.forecast is None:
            sys.exit("ERROR: -estwin requires -forecast H")
        rec_path = base + ".recursive"
        report.write_recursive(model, args.estwin, args.forecast, rec_path)
        print("Recursive forecasts written to %s" % rec_path)

    if args.volexp is not None or args.volmov is not None:
        from . import volatility
        res = model.result["residuals"]
        if args.volexp is not None:
            alpha = float(args.volexp[0]) if len(args.volexp) >= 1 else volatility.DEFAULT_ALPHA
            window = int(args.volexp[1]) if len(args.volexp) >= 2 else volatility.DEFAULT_WINDOW
            ve_path = base + ".volexp"
            phi, thr = volatility.write_volexp(ve_path, res, model.result["sigma"],
                                               alpha, window)
            with open(out_path, "a") as f:
                f.write(volatility.volexp_info_section(alpha, thr, phi))
            print("Exponential volatility series written to %s" % ve_path)
        if args.volmov is not None:
            window = int(args.volmov[0]) if len(args.volmov) >= 1 else volatility.DEFAULT_WINDOW
            vm_path = base + ".volmov"
            volatility.write_volmov(vm_path, res, window)
            print("Moving-window volatility series written to %s" % vm_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
