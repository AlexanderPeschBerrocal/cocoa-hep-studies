# cocoa-hep-studies

Physics workflow notes for:
- COCOA simulation
- ROOT inspection
- Python plotting
- Phoenix frontend

---

# Quick Navigation

1. [Generate ROOT files with COCOA](#1-generate-root-files-with-cocoa)
2. [Inspect ROOT files with ROOT](#2-inspect-root-files-with-root)
3. [Create nicer Python plots](#3-create-nicer-python-plots)
4. [Run Phoenix frontend](#4-run-phoenix-frontend)

---

# Cheat Sheet

## Activate environments

### Python plotting environment

```bash
micromamba activate cocoa_py
```

### Phoenix environment

```bash
micromamba activate phoenix
```

---

## Start COCOA container

```bash
apptainer shell \
  --bind ~/cocoa_project:/work \
  ~/containers/cocoa-hep.sif
```

Exit container:

```bash
exit
```

---

## Common ROOT commands

```cpp
.ls
Out_Tree->Print()
```

Quit ROOT:

```cpp
.q
```

---

# 1. Generate ROOT Files with COCOA

## Create run directory

```bash
mkdir -p ~/cocoa_project/runs/my_run
```

---

## Start container

```bash
apptainer shell \
  --bind ~/cocoa_project:/work \
  ~/containers/cocoa-hep.sif
```

---

## Go to COCOA directory

```bash
cd /root/COCOA/COCOA
```

---

## Verify COCOA installation

```bash
../build/COCOA -h
```

---

## Run simulation

```bash
../build/COCOA \
  --macro macro/Pythia8/ttbar.in \
  --config config/config_default.json \
  --output /work/runs/my_run/output.root \
  --seed 1 \
  --nevents 1000
```

---

## Verify output

```bash
ls -lh /work/runs/my_run
```

Expected output:

```text
output.root
```

---

## Exit container

```bash
exit
```

ROOT file will now exist on host system:

```text
~/cocoa_project/runs/my_run/output.root
```

---

# 2. Inspect ROOT Files with ROOT

## Start container

```bash
apptainer shell \
  --bind ~/cocoa_project:/work \
  ~/containers/cocoa-hep.sif
```

---

## Open ROOT file

```bash
root -l /work/runs/my_run/output.root
```

---

## Inspect contents

```cpp
.ls
Out_Tree->Print()
```

---

## XY conversion plot

```cpp
TCanvas *c1 = new TCanvas("c1","c1",800,600);

Out_Tree->Draw(
  "conv_el_prod_y:conv_el_prod_x"
);

c1->SaveAs(
  "/work/runs/my_run/conv_xy_root.png"
);
```

---

## RZ conversion plot

```cpp
TCanvas *c2 = new TCanvas("c2","c2",800,600);

Out_Tree->Draw(
  "conv_el_prod_z:sqrt(conv_el_prod_x*conv_el_prod_x + conv_el_prod_y*conv_el_prod_y)"
);

c2->SaveAs(
  "/work/runs/my_run/conv_rz_root.png"
);
```

---

## Exit ROOT

```cpp
.q
```

---

## Exit container

```bash
exit
```

---

## Open plots in VS Code

```text
~/cocoa_project/runs/my_run/conv_xy_root.png
~/cocoa_project/runs/my_run/conv_rz_root.png
```

---

# 3. Create Nicer Python Plots

## Activate Python environment

```bash
micromamba activate cocoa_py
```

---

## Run plotting script

```bash
python ~/cocoa_project/scripts/inspect_cocoa_root.py \
  ~/cocoa_project/runs/my_run/output.root \
  --outdir ~/cocoa_project/runs/my_run/python_plots
```

---

## Verify output

```bash
ls -lh ~/cocoa_project/runs/my_run/python_plots
```

---

## Open plots

Open generated PNGs directly in VS Code.

---

# 4. Run Phoenix Frontend

## Activate environment

```bash
micromamba activate phoenix
```

---

## Go to Phoenix directory

```bash
cd ~/software/phoenix
```

---

## Start Phoenix

```bash
yarn start
```

Wait for successful compilation.

---

## Forward port in VS Code

```text
Ports → Forward Port → 4200
```

---

## Open in browser

```text
http://localhost:4200
```

---

## Stop Phoenix

```text
Ctrl+C
```

---

# 5. Convert COCOA ROOT to Phoenix JSON

Start Apptainer:

apptainer shell --bind ~/cocoa_project:/work ~/containers/cocoa-hep.sif

Go to the converter:

cd /root/COCOA/COCOA/phoenix/event

Run the converter using the host micromamba Python:

/.automount/home/home__home1/institut_3a/pesch/micromamba/envs/cocoa_py/bin/python \
dump_phoenix_eventdata.py \
-i /work/runs/my_run/output.root \
-o /work/runs/my_run/cocoa_event_1.json \
-n 1

Check:

ls -lh /work/runs/my_run/cocoa_event_1.json

Exit:

exit

# 6. Open COCOA event with Phoenix

Start Phoenix

load event json and optionally geometry files
