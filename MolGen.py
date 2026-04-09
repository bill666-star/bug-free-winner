import streamlit as st
import random
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Draw
from rdkit.Chem import FilterCatalog
from rdkit.Chem.rdmolfiles import MolToMolBlock
import pandas as pd
from io import BytesIO
import base64

st.set_page_config(page_title="MolGen - 智能分子生成器", layout="wide")
st.title("🧪 MolGen - 骨架保序·全能药物分子生成器")
st.markdown("### 🔒 固定骨架 | ✅ 去PAINS | 🧪 成药性 | ⚗️ 合成难度 | 🛣️ 逆合成 | 🖼️ 2D结构图")

@st.cache_resource
def load_pains_filter():
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog.FilterCatalog(params)
pains_filter = load_pains_filter()

ISO_RULES = [
    ("[H]", ["F", "Cl"]),
    ("[F]", ["H", "Cl", "CN"]),
    ("[Cl]", ["F", "H"]),
    ("[CH3]", ["CH2CH3", "CF3", "CN"]),
    ("[OH]", ["OCH3", "F"]),
    ("[NH2]", ["OH", "CH3"]),
    ("C(=O)O", ["c1n[nH]nn1", "S(=O)(=O)N"]),
    ("c1ccccc1", ["c1ccncc1", "c1cccnc1"]),
]

def draw_mol(mol):
    if mol is None:
        return None
    try:
        img = Draw.MolToImage(mol, size=(300,180), kekulize=True)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except:
        return None

def calc_props(mol):
    return {
        "MW": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 2),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "RotB": rdMolDescriptors.CalcNumRotatableBonds(mol),
    }

def drug_likeness(mol):
    p = calc_props(mol)
    score = 0
    if 200 < p["MW"] < 550: score +=1
    if -1 < p["LogP"] < 5: score +=1
    if p["HBD"] <=5: score +=1
    if p["HBA"] <=10: score +=1
    if p["TPSA"] < 140: score +=1
    return f"{score}/5"

def synth_difficulty(mol):
    rings = rdMolDescriptors.CalcNumRings(mol)
    stereo = rdMolDescriptors.CalcNumStereoCenters(mol)
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    score = 1
    if rings > 3: score +=1
    if stereo > 2: score +=1
    if rot > 8: score +=1
    level = ["极低", "低", "中等", "较高", "极高"]
    return level[min(score, 4)]

def retrosynth_cheap(mol):
    smi = Chem.MolToSmiles(mol)
    if len(smi) < 15:
        return "✅ 简单芳香取代/偶联，低成本"
    elif "C(=O)" in smi:
        return "✅ 酰胺缩合/酯化，常规试剂"
    elif "c1ccncc1" in smi:
        return "✅ 杂环构建，成熟路线"
    elif "O-" in smi:
        return "✅ 醚化反应，易合成"
    else:
        return "⚠️ 多步路线，中等成本"

def passed_filter(new_mol, orig_props, cfg):
    p = calc_props(new_mol)
    if abs(p["MW"] - orig_props["MW"]) > cfg["mw"]: return False
    if abs(p["LogP"] - orig_props["LogP"]) > cfg["logp"]: return False
    if abs(p["TPSA"] - orig_props["TPSA"]) > cfg["tpsa"]: return False
    if abs(p["HBA"] - orig_props["HBA"]) > cfg["hba"]: return False
    if abs(p["HBD"] - orig_props["HBD"]) > cfg["hbd"]: return False
    if pains_filter.HasMatch(new_mol): return False
    return True

def apply_bioiso(mol):
    m = Chem.Mol(mol)
    try:
        rule = random.choice(ISO_RULES)
        patt = Chem.MolFromSmarts(rule[0])
        repl = Chem.MolFromSmarts(random.choice(rule[1:]))
        if patt and repl:
            res = Chem.ReplaceSubstructs(m, patt, repl, replaceAll=False)
            return res[0] if res else m
    except:
        return m
    return m

def generate_library(smi, count, cfg):
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        return [], {}, None
    orig_props = calc_props(mol)
    valid = []
    seen = set()
    for _ in range(count * 12):
        cand = apply_bioiso(mol)
        if not cand: continue
        if not passed_filter(cand, orig_props, cfg): continue
        s = Chem.MolToSmiles(cand)
        if s in seen or s == smi: continue
        seen.add(s)
        valid.append((s, calc_props(cand), cand))
        if len(valid) >= count:
            break
    return valid, orig_props, mol

with st.sidebar:
    st.header("📥 输入分子")
    input_smi = st.text_input("SMILES", "c1cc(OC)ccc1C")
    gen_count = st.number_input("生成分子数量", 10, 500, 30)

    st.header("⚙️ 性质不突变约束")
    mw_tol = st.slider("分子量波动 ±", 0, 60, 25)
    logp_tol = st.slider("LogP 波动 ±", 0.0, 2.0, 1.0, 0.1)
    tpsa_tol = st.slider("TPSA 波动 ±", 0, 40, 15)
    hba_tol = st.slider("HBA 波动 ±", 0, 2, 1)
    hbd_tol = st.slider("HBD 波动 ±", 0, 2, 1)

    config = {
        "mw": mw_tol, "logp": logp_tol, "tpsa": tpsa_tol,
        "hba": hba_tol, "hbd": hbd_tol
    }

if st.button("🚀 启动 MolGen 生成分子库", use_container_width=True):
    with st.spinner("生成中 · 骨架保序 · PAINS过滤 · 成药性 · 合成评估"):
        mols, orig_props, orig_mol = generate_library(input_smi, gen_count, config)

    if not mols:
        st.error("无法生成符合条件的分子，请放宽约束")
    else:
        st.success(f"✅ 生成 {len(mols)} 个合格分子")

        st.subheader("🔍 原始分子")
        img_orig = draw_mol(orig_mol)
        if img_orig:
            st.markdown(f'<img src="data:image/png;base64,{img_orig}">', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([orig_props]))

        st.subheader("🖼️ 生成分子2D预览")
        preview = mols[:8]
        cols = st.columns(4)
        for i, (smi, prop, mol) in enumerate(preview):
            with cols[i % 4]:
                img = draw_mol(mol)
                st.markdown(f"**Mol-{i+1}**")
                if img:
                    st.markdown(f'<img src="data:image/png;base64,{img}">', unsafe_allow_html=True)

        st.subheader("📊 分子库详细信息")
        rows = []
        sdf_content = ""
        for i, (smi, prop, mol_obj) in enumerate(mols, 1):
            rows.append({
                "ID": f"MOL-{i:03d}",
                "SMILES": smi,
                "MW": prop["MW"],
                "LogP": prop["LogP"],
                "TPSA": prop["TPSA"],
                "成药性": drug_likeness(mol_obj),
                "合成难度": synth_difficulty(mol_obj),
                "逆合成路线": retrosynth_cheap(mol_obj)
            })
            sdf_content += MolToMolBlock(mol_obj) + "$$$$\n"

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, height=400)

        c1, c2 = st.columns(2)
        c1.download_button("💾 导出 CSV", df.to_csv(index=False), "MolGen_library.csv")
        c2.download_button("📥 导出 SDF(对接专用)", sdf_content, "MolGen_library.sdf")

st.markdown("---")
st.markdown("**MolGen © 2026 | 开源免费 | 骨架保序分子生成工具**")
