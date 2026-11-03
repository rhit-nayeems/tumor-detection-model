import argparse, json, os

import numpy as np, torch, SimpleITK as sitk

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor



ap = argparse.ArgumentParser()

ap.add_argument("--chunk", type=int, required=True)

ap.add_argument("--num-chunks", type=int, required=True)

ap.add_argument("--out", required=True)

args = ap.parse_args()



AP = os.environ["AP"]

RAW = f"{os.environ['nnUNet_raw']}/Dataset200_autoPET_lesions"

MODEL = (f"{os.environ['nnUNet_results']}/Dataset200_autoPET_lesions/"

         "autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres")

os.makedirs(args.out, exist_ok=True)

out_file = f"{args.out}/chunk_{args.chunk:03d}.npz"

if os.path.exists(out_file):

    print("Chunk already done:", out_file); raise SystemExit



# This chunk's share of all cases (train + val + test)

split = json.load(open(f"{AP}/split_60_20_20.json"))

cases = sorted((c, s) for s in split for c in split[s])

mine = np.array_split(np.arange(len(cases)), args.num_chunks)[args.chunk]



# Load the trained network and keep only the encoder

ckpt = next(n for n in ("checkpoint_final.pth", "checkpoint_best.pth", "checkpoint_latest.pth")

            if os.path.exists(f"{MODEL}/fold_0/{n}"))

print("Using", ckpt, flush=True)

pred = nnUNetPredictor(device=torch.device("cuda"))

pred.initialize_from_trained_model_folder(MODEL, use_folds=(0,), checkpoint_name=ckpt)

pred.network.load_state_dict(pred.list_of_parameters[0])

enc = pred.network.encoder.cuda().eval()

pre = pred.configuration_manager.preprocessor_class(verbose=False)

P = pred.configuration_manager.patch_size



def starts(dim, p):

    if dim <= p: return [0]

    s = list(range(0, dim - p + 1, p))

    return s if s[-1] == dim - p else s + [dim - p]



@torch.no_grad()

def encode(data):

    x = torch.from_numpy(data).float()

    pad = [max(0, p - d) for p, d in zip(P, x.shape[1:])]

    x = torch.nn.functional.pad(x, [v for pd in reversed(pad) for v in (0, pd)])

    feats = []

    for z in starts(x.shape[1], P[0]):

        for y in starts(x.shape[2], P[1]):

            for w in starts(x.shape[3], P[2]):

                t = x[None, :, z:z+P[0], y:y+P[1], w:w+P[2]].cuda()

                with torch.autocast("cuda", dtype=torch.float16):

                    b = enc(t)[-1]                       # deepest encoder features

                feats.append(b.float().mean(dim=(2, 3, 4))[0].cpu())

    f = torch.stack(feats)

    return torch.cat([f.mean(0), f.max(0).values]).numpy()   # one vector per scan



ids, splits, lesion, feats = [], [], [], []

for n, i in enumerate(mine):

    c, s = cases[i]

    d = "Ts" if s == "test" else "Tr"

    imgs = [f"{RAW}/images{d}/{c}_0000.nii.gz", f"{RAW}/images{d}/{c}_0001.nii.gz"]

    data, _, _ = pre.run_case(imgs, None, pred.plans_manager, pred.configuration_manager, pred.dataset_json)

    feats.append(encode(data))

    lab = sitk.GetArrayFromImage(sitk.ReadImage(f"{RAW}/labels{d}/{c}.nii.gz"))

    lesion.append(int(lab.any())); ids.append(c); splits.append(s)

    print(f"[{n+1}/{len(mine)}] {c}", flush=True)



np.savez(out_file, case_ids=np.array(ids), split=np.array(splits),

         tracer=np.array([c.split("_")[0] for c in ids]),

         lesion=np.array(lesion), features=np.stack(feats))

print("Saved", out_file)
