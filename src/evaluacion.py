import json
import sys
from pathlib import Path

import numpy as np

from src.modelo import EmbeddingExtractor, EstrategiaEmbedding
from src.chunking import similitud_coseno, ratio_palabras_nuevas

VENTANA = 3


def cargar_dialogos(ruta: str, n: int = 20):
    with open(ruta, encoding="utf-8") as f:
        data = json.load(f)
    return data["dial_data"]["dialseg711"][:n]


def ground_truth_bounds(turns: list) -> set[int]:
    return {i for i, t in enumerate(turns) if t["segmentation_label"] == 1}


def evaluar_embed(scores, label: str):
    best = {"th": 0, "f1": 0}
    for th in np.arange(0.01, 1.0, 0.01):
        by_dial = {}
        for s in scores:
            if s["dial"] not in by_dial:
                by_dial[s["dial"]] = []
            by_dial[s["dial"]].append(s)

        tp_all = fp_all = fn_all = 0
        for ds in by_dial.values():
            reales = sorted(ds[0]["reales"])
            preds = [s["pos"] for s in ds if s["sim"] < th]

            rc = set()
            for rb in reales:
                for dp in preds:
                    if abs(dp - rb) <= 1:
                        rc.add(rb)
                        break
            fp_count = 0
            for dp in preds:
                if not any(abs(dp - rb) <= 1 for rb in reales):
                    fp_count += 1

            tp_all += len(rc)
            fp_all += fp_count
            fn_all += len(reales) - len(rc)

        p = tp_all / (tp_all + fp_all) if tp_all + fp_all else 0
        r = tp_all / (tp_all + fn_all) if tp_all + fn_all else 0
        f1 = 2 * p * r / (p + r) if p + r else 0
        if f1 > best["f1"]:
            best = {"th": round(th, 2), "f1": round(f1, 3), "p": round(p, 3), "r": round(r, 3), "tp": tp_all, "fp": fp_all, "fn": fn_all}
    print(f"  {label}: th={best['th']} | TP={best['tp']} FP={best['fp']} FN={best['fn']} | P={best['p']} R={best['r']} F1={best['f1']}")


def evaluar_and(scores, label: str, campo_extra: str):
    best = {"th_e": 0, "th_x": 0, "f1": 0}
    for th_e in np.arange(0.01, 1.0, 0.01):
        for th_x in np.arange(0.01, 1.0, 0.01):
            by_dial = {}
            for s in scores:
                if s["dial"] not in by_dial:
                    by_dial[s["dial"]] = []
                by_dial[s["dial"]].append(s)

            tp_all = fp_all = fn_all = 0
            for ds in by_dial.values():
                reales = sorted(ds[0]["reales"])
                preds = []
                for s in ds:
                    if s["sim"] < th_e and s[campo_extra] > th_x:
                        preds.append(s["pos"])

                rc = set()
                for rb in reales:
                    for dp in preds:
                        if abs(dp - rb) <= 1:
                            rc.add(rb)
                            break
                fp_count = 0
                for dp in preds:
                    if not any(abs(dp - rb) <= 1 for rb in reales):
                        fp_count += 1

                tp_all += len(rc)
                fp_all += fp_count
                fn_all += len(reales) - len(rc)

            p = tp_all / (tp_all + fp_all) if tp_all + fp_all else 0
            r = tp_all / (tp_all + fn_all) if tp_all + fn_all else 0
            f1 = 2 * p * r / (p + r) if p + r else 0
            if f1 > best["f1"]:
                best = {"th_e": round(th_e, 2), "th_x": round(th_x, 2), "f1": round(f1, 3), "p": round(p, 3), "r": round(r, 3), "tp": tp_all, "fp": fp_all, "fn": fn_all}
    print(f"  {label}: th_e={best['th_e']} th_x={best['th_x']} | TP={best['tp']} FP={best['fp']} FN={best['fn']} | P={best['p']} R={best['r']} F1={best['f1']}")


def main():
    ruta_json = Path("data/dialseg711_test.json")
    n_dialogos = int(sys.argv[1]) if len(sys.argv) > 1 else 20

    if not ruta_json.exists():
        import requests
        resp = requests.get(
            "https://huggingface.co/datasets/Coldog2333/dialseg711/resolve/main/test.json"
        )
        ruta_json.write_text(resp.text, encoding="utf-8")

    print(f"Cargando {n_dialogos} dialogos...", flush=True)
    dialogos = cargar_dialogos(str(ruta_json), n_dialogos)

    print("Inicializando modelo...", flush=True)
    extractor = EmbeddingExtractor()

    print("Extrayendo embeddings...", flush=True)
    ddata = []
    for idx, dial in enumerate(dialogos):
        utterances = [t["utterance"] for t in dial["turns"]]
        bounds = ground_truth_bounds(dial["turns"])
        embs = [extractor.extraer_embedding(u, EstrategiaEmbedding.MEDIA_GLOBAL) for u in utterances]
        ddata.append({
            "id": dial["dial_id"], "embs": np.array(embs), "bounds": bounds,
            "n": len(utterances), "utterances": utterances,
        })
        print(f"  [{idx+1:2d}/{n_dialogos}] {dial['dial_id']}: {len(utterances)} turns", flush=True)

    for d in ddata:
        ng = (d["n"] + VENTANA - 1) // VENTANA
        d["ng"] = ng
        d["cents"] = [d["embs"][g*VENTANA:min(g*VENTANA+VENTANA, d["n"])].mean(axis=0) for g in range(ng)]
        d["window_texts"] = [
            " ".join(d["utterances"][g*VENTANA:min(g*VENTANA+VENTANA, d["n"])])
            for g in range(ng)
        ]

    # Senal 1: embedding sola (baseline)
    scores_embed = []
    for d in ddata:
        for g in range(d["ng"] - 1):
            sm = similitud_coseno(d["cents"][g], d["cents"][g + 1])
            scores_embed.append({"sim": sm, "pos": (g + 1) * VENTANA - 1, "dial": d["id"], "reales": d["bounds"]})
    evaluar_embed(scores_embed, "Embedding sola (ventana=3)")

    # Senal 2: embedding + sustantivos nuevos (AND) -- configuracion ganadora
    scores_nouns = []
    for d in ddata:
        for g in range(d["ng"] - 1):
            sm = similitud_coseno(d["cents"][g], d["cents"][g + 1])
            rn = ratio_palabras_nuevas(d["window_texts"][g], d["window_texts"][g + 1])
            scores_nouns.append({
                "sim": sm, "ratio_nuevas": rn, "pos": (g + 1) * VENTANA - 1,
                "dial": d["id"], "reales": d["bounds"],
            })
    evaluar_and(scores_nouns, "Embedding + sustantivos nuevos (AND)", "ratio_nuevas")


if __name__ == "__main__":
    main()
