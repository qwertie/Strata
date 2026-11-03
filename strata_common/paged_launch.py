"""Adapts saved server settings for the opt-in page-pool engines."""
from pathlib import Path


def is_paged_engine(cfg: dict) -> bool:
    """Recognizes the two page-pool programs selected by the saved executable path."""
    return get_engine_name(cfg) in ("strata-q38", "strata-ds41")


def translate_args(cfg: dict, args: list[str]) -> list[str]:
    """Keeps supported options and reports settings with no page-pool equivalent."""
    deepseek = get_engine_name(cfg) == "strata-ds41"
    common = set("""pack max-context batch expert-cache expert-profile layer-split split-device
        adapt-every adapt-swaps adapt-min-gain adapt-decay tokens-min tokens-max gpu-mib ram-mib ssd-pause-mib
        rows lookahead concurrent pause-dir copy-back drafter remote-experts""".split())
    model = set(("""vision eos gpus reserve-mib cpu-threads dspark dspark-conf experts dspark-block prefill-chunk threads"""
                 if deepseek else """eos-ids native ple-gguf kv prefill mtp spec-min-p gpu pcie-frac pool-cores
                 aux-cpus pool-affinity vram-reserve-mib vram-reserve-later-mib""").split())
    switches = {"serve", "trim-stage-weights"} | ({"no-expert-cache"} if deepseek else
                 {"vision", "no-ple", "no-prefill-borrow", "no-host-worker", "no-ple-prefetch", "stop-eos"})
    aliases = {"--slots": "--batch"}
    aliases["--gpu" if deepseek else "--gpus"] = "--gpus" if deepseek else "--gpu"
    if deepseek:
        aliases["--vram-reserve-mib"] = "--reserve-mib"
    else:
        aliases["--resident-budget-gib"] = "--ram-mib"
        aliases["--eos"] = "--eos-ids"
    supported = common | model | switches
    out, omitted = [], []
    i = 0
    while i < len(args):
        flag = str(args[i])
        end = i + 1
        while end < len(args) and not str(args[end]).startswith("--"):
            end += 1
        values = args[i + 1:end]
        target = aliases.get(flag, flag)
        name = target.removeprefix("--")
        if flag == "--resident-budget-gib" and "--ram-mib" in args:
            omitted.extend([flag, *values])
        elif name in supported:
            if name in switches:
                out.append(target)
            elif values:
                if flag == "--resident-budget-gib":
                    values = [str(int(float(values[0]) * 1024))]
                out.extend([target, *values])
            elif name == "vision" and deepseek:
                out.extend([target, "on"])
            else:
                raise ValueError(f"{flag} needs a value")
        else:
            omitted.extend([flag, *values])
        i = end
    if cfg.get("mtp") is False or cfg.get("drafting") == "off":
        out += ["--drafter", "off", "--lookahead", "0"]
        if deepseek:
            out += ["--dspark", "none"]
    ignored = [key for key in ("vram_elastic", "split_skip_if_fits", "batchMtp") if cfg.get(key)]
    if cfg.get("drafting") == "dflash2":
        ignored.append("drafting=dflash2")
    if ignored:
        print("[strata] paged engine: leaving out unsupported config settings: " + ", ".join(ignored), flush=True)
    if omitted:
        print("[strata] paged engine: leaving out unsupported settings: " + " ".join(omitted), flush=True)
    return out


def get_engine_name(cfg: dict) -> str:
    """Gets the executable basename from config paths saved on either operating system."""
    return Path(str(cfg.get("exe", "")).replace("\\", "/")).stem.lower()
