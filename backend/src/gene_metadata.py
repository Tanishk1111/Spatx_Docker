"""
Gene Metadata for 50-Gene Breast Cancer Panel
Contains functional classification, clinical significance, and correlation scores
"""

# Gene categories for grouping
GENE_CATEGORIES = {
    "Hormone Receptors & Signaling": ["ESR1", "ERBB2", "GATA3", "KLF5"],
    "Growth Factor Receptors & Proliferation": ["KIT", "OXTR", "TOP2A"],
    "Basal/Myoepithelial Markers": ["KRT5", "KRT14", "KRT6B"],
    "Extracellular Matrix & Metastasis": ["MMP1", "MMP12"],
    "Immune Cell Markers": ["CD3E", "CCR7", "IL2RG", "IL7R", "MS4A1", "PTPRC", "SLAMF7", "TCL1A", "TRAC"],
    "Immune-Related Cell Adhesion": ["CEACAM6", "CEACAM8"],
    "Lipid Metabolism": ["FASN", "SCD", "ADIPOQ"],
    "Water & Small Molecule Transport": ["AQP1", "AQP3"],
    "Secreted & Glandular Markers": ["PIGR", "MUC6", "SCGB2A1", "PTGDS", "TPSAB1", "SERPINA3", "PTN"],
    "Cytoskeletal & Contractile Proteins": ["MYH11", "MYLK", "MYBPC1"],
    "Signaling & Regulatory Proteins": ["SFRP1", "CLIC6", "CYTIP", "OPRPN", "SERHL2", "TENT5C", "ANKRD30A", "ABCC11"],
    "Structural & Adhesion Proteins": ["DST", "VWF", "TACSTD2"],
    "Enzyme & Metabolic Function": ["ADH1B"]
}

# Gene information dictionary
GENE_INFO = {
    # Hormone Receptors & Signaling
    "ESR1": {
        "full_name": "Estrogen Receptor 1",
        "function": "Encodes estrogen receptor alpha (ERα), a nuclear hormone receptor crucial for estrogen signaling in breast tissue. Activating mutations in ESR1 ligand-binding domain drive resistance to hormonal therapy in ER+ breast cancer.",
        "clinical_significance": "ESR1 mutations are key biomarkers for stratifying ER+/HER2- advanced breast cancer and predicting response to endocrine therapy. Testing for ESR1 mutations guides treatment decisions in hormone receptor-positive metastatic breast cancer.",
        "pearson_correlation": 0.88
    },
    "ERBB2": {
        "full_name": "HER2 (Human Epidermal Growth Factor Receptor 2)",
        "function": "Encodes HER2, a tyrosine kinase receptor that promotes cell proliferation and survival. Amplification/overexpression drives aggressive tumor growth and is found in approximately 15-20% of breast cancers.",
        "clinical_significance": "HER2 status determines eligibility for targeted therapies like trastuzumab, pertuzumab, and T-DM1. Routinely tested in all invasive breast cancers as part of standard pathological evaluation. HER2-positive breast cancers are more aggressive but respond well to targeted therapies.",
        "pearson_correlation": 0.91
    },
    "GATA3": {
        "full_name": "GATA Binding Protein 3",
        "function": "Transcription factor essential for luminal breast epithelium development and maintenance. Co-regulates genes with estrogen receptor and modulates hormone responsiveness. Required for mammary gland morphogenesis and differentiation.",
        "clinical_significance": "High GATA3 expression correlates with favorable prognosis, ER positivity, and better response to hormonal therapy. Used as a diagnostic marker for breast cancer origin in metastatic disease. GATA3 frameshift mutations can paradoxically promote tumor growth despite being a luminal marker.",
        "pearson_correlation": 0.91
    },
    "KLF5": {
        "full_name": "Kruppel-Like Factor 5",
        "function": "Transcription factor involved in cell proliferation, differentiation, and epithelial-mesenchymal transition. Plays dual roles as both tumor suppressor and oncogene depending on cellular context and breast cancer subtype.",
        "clinical_significance": "Expression patterns vary across breast cancer subtypes and influence therapeutic response. Associated with stem cell-like properties in certain breast cancer contexts. Can promote or inhibit tumorigenesis depending on molecular context.",
        "pearson_correlation": 0.86
    },
    
    # Growth Factor Receptors & Proliferation
    "KIT": {
        "full_name": "KIT Proto-Oncogene, Receptor Tyrosine Kinase",
        "function": "Encodes receptor tyrosine kinase for stem cell factor (SCF), involved in cell survival, proliferation, and differentiation. Essential for development of melanocytes, hematopoietic cells, and germ cells.",
        "clinical_significance": "KIT expression varies across breast cancer subtypes. While activating mutations are rare in breast cancer compared to gastrointestinal stromal tumors, KIT may represent a therapeutic target in specific tumor contexts with pathway activation.",
        "pearson_correlation": 0.37
    },
    "OXTR": {
        "full_name": "Oxytocin Receptor",
        "function": "G-protein coupled receptor for oxytocin hormone, involved in lactation, uterine contractions, and social bonding behaviors. May have tumor suppressive effects in breast tissue through anti-proliferative signaling.",
        "clinical_significance": "Higher expression associated with better prognosis in breast cancer. May have protective effects against breast cancer development, particularly related to breastfeeding benefits. Oxytocin signaling may inhibit tumor cell migration and invasion.",
        "pearson_correlation": 0.68
    },
    "TOP2A": {
        "full_name": "Topoisomerase II Alpha",
        "function": "Essential enzyme for DNA replication and chromosome segregation. Catalyzes transient double-strand breaks to relieve topological stress during DNA replication, transcription, and chromosome condensation.",
        "clinical_significance": "Co-amplified with HER2 in 30-40% of HER2-positive breast cancers. Predicts response to anthracycline chemotherapy (doxorubicin, epirubicin). Serves as proliferation marker with high expression indicating rapidly dividing cells. Used to assess proliferative index of tumors.",
        "pearson_correlation": 0.80
    },
    
    # Basal/Myoepithelial Markers
    "KRT5": {
        "full_name": "Keratin 5",
        "function": "Type II intermediate filament protein specifically expressed in basal epithelial cells and myoepithelial cells. Provides structural integrity to basal cell layer and forms obligate heterodimers with keratin 14 to create intermediate filament networks.",
        "clinical_significance": "Defining marker of basal-like breast cancer subtype. KRT5 overexpression strongly associates with BRCA1-mutated tumors, triple-negative breast cancer, and poor prognosis. Used in molecular classification schemes to distinguish basal-like from other TNBC subtypes.",
        "pearson_correlation": 0.66
    },
    "KRT14": {
        "full_name": "Keratin 14",
        "function": "Type I keratin that pairs with KRT5 in basal epithelial cells and myoepithelial cells. Forms intermediate filaments crucial for mechanical stability and cellular architecture in stratified epithelia.",
        "clinical_significance": "Co-expressed with KRT5 in basal-like breast cancers. Used to classify basal phenotype and distinguish from luminal subtypes. Indicates stem cell-like properties and is associated with more aggressive tumor behavior. Part of the basal cytokeratin panel for immunohistochemistry.",
        "pearson_correlation": 0.57
    },
    "KRT6B": {
        "full_name": "Keratin 6B",
        "function": "Type II keratin expressed in proliferative epithelial cells, particularly during wound healing and hyperproliferative conditions. Upregulated in response to cellular stress and tissue injury.",
        "clinical_significance": "May indicate proliferative and invasive potential in breast tumors. Associated with more aggressive tumor behavior and poor differentiation. Expression suggests activated epithelial state with enhanced migratory and proliferative capacity.",
        "pearson_correlation": 0.52
    },
    
    # Extracellular Matrix & Metastasis
    "MMP1": {
        "full_name": "Matrix Metalloproteinase 1 (Collagenase-1)",
        "function": "Collagenase that degrades type I, II, and III collagens in extracellular matrix. Essential for normal tissue remodeling during wound healing and development, but facilitates tumor invasion when overexpressed.",
        "clinical_significance": "Overexpression promotes metastasis by degrading basement membrane and facilitating tumor cell invasion through stromal tissue. Associated with poor prognosis, increased risk of distant metastases, and reduced disease-free survival. Produced by both tumor cells and cancer-associated fibroblasts in the tumor microenvironment.",
        "pearson_correlation": 0.39
    },
    "MMP12": {
        "full_name": "Matrix Metalloproteinase 12 (Macrophage Elastase)",
        "function": "Metalloelastase that degrades elastin and other ECM components including fibronectin, laminin, and proteoglycans. Involved in tissue remodeling, inflammation, and angiogenesis. Primarily secreted by macrophages but also expressed by tumor cells.",
        "clinical_significance": "Associated with metastatic potential and tumor microenvironment remodeling. May facilitate immune cell infiltration and angiogenesis while promoting invasive capacity. Expression correlates with advanced stage disease and poor outcomes in multiple cancer types including breast cancer.",
        "pearson_correlation": 0.17
    },
    
    # Immune Cell Markers
    "CD3E": {
        "full_name": "CD3 Epsilon",
        "function": "Essential component of T-cell receptor complex required for T-cell activation and intracellular signaling. Expressed on all mature T cells and necessary for TCR surface expression and signal transduction.",
        "clinical_significance": "Marker of tumor-infiltrating T lymphocytes (TILs). High TIL levels predict better response to chemotherapy and immunotherapy, particularly in triple-negative breast cancer. CD3+ TIL density is an independent prognostic factor for improved survival.",
        "pearson_correlation": 0.83
    },
    "CCR7": {
        "full_name": "C-C Motif Chemokine Receptor 7",
        "function": "Chemokine receptor that guides T cells and dendritic cells to lymph nodes via CCL19 and CCL21 ligands. Critical for adaptive immune response initiation and lymphocyte homing to secondary lymphoid organs.",
        "clinical_significance": "Expression on tumor cells may facilitate lymph node metastasis by exploiting normal lymphocyte trafficking pathways. On immune cells, indicates capacity for trafficking to lymphoid organs and immune activation. Dual role depending on cell type.",
        "pearson_correlation": 0.81
    },
    "IL2RG": {
        "full_name": "Interleukin 2 Receptor Gamma Chain (CD132)",
        "function": "Common gamma chain shared by receptors for IL-2, IL-4, IL-7, IL-9, IL-15, and IL-21. Critical for lymphocyte development, proliferation, and survival. Mutations cause X-linked severe combined immunodeficiency (SCID).",
        "clinical_significance": "Marker of immune cell activation and proliferation. Essential for T cell and NK cell function in antitumor immunity. High expression indicates active lymphocyte signaling and proliferation within tumor microenvironment.",
        "pearson_correlation": 0.81
    },
    "IL7R": {
        "full_name": "Interleukin 7 Receptor Alpha Chain (CD127)",
        "function": "Receptor for IL-7, essential for T and B cell development, homeostasis, and survival. Critical for maintaining naive and memory T lymphocyte populations. Pairs with IL2RG to form functional IL-7 receptor.",
        "clinical_significance": "Expressed on lymphocytes infiltrating tumors, indicating ongoing adaptive immune response. Associated with better prognosis when highly expressed. Marks T cell populations capable of sustained antitumor responses.",
        "pearson_correlation": 0.85
    },
    "MS4A1": {
        "full_name": "Membrane Spanning 4-Domains A1 (CD20)",
        "function": "B-cell surface antigen expressed throughout B-cell development from pre-B cells to mature B cells, but not on plasma cells. Functions in B-cell activation, proliferation, and calcium signaling. Target of rituximab and other anti-CD20 antibodies.",
        "clinical_significance": "Marker of B-cell infiltration in tumor microenvironment. B-cell presence may indicate tertiary lymphoid structures (TLS), which are associated with better prognosis and immunotherapy response. CD20+ B cells contribute to adaptive antitumor immunity.",
        "pearson_correlation": 0.76
    },
    "PTPRC": {
        "full_name": "Protein Tyrosine Phosphatase Receptor Type C (CD45)",
        "function": "Protein tyrosine phosphatase receptor expressed on all hematopoietic cells except erythrocytes and platelets. Essential for immune cell signaling, activation, and development. Regulates Src family kinases in antigen receptor signaling.",
        "clinical_significance": "Pan-leukocyte marker used to quantify total immune cell infiltration in tumors. High expression indicates 'hot' immune microenvironment that may respond better to immunotherapy. Used to distinguish immune cells from tumor cells in pathology.",
        "pearson_correlation": 0.78
    },
    "SLAMF7": {
        "full_name": "SLAM Family Member 7 (CD319, CS1)",
        "function": "Cell surface receptor on natural killer cells, activated T cells, and plasma cells. Involved in immune cell activation, cytotoxicity, and adhesion. Elotuzumab (anti-SLAMF7) targets myeloma cells.",
        "clinical_significance": "Marker of NK cell and plasma cell infiltration. High expression may indicate active antitumor immune response with NK cell-mediated cytotoxicity. NK cells play important role in antibody-dependent cellular cytotoxicity (ADCC) in HER2+ breast cancer treated with trastuzumab.",
        "pearson_correlation": 0.77
    },
    "TCL1A": {
        "full_name": "T-Cell Leukemia/Lymphoma 1A",
        "function": "Oncogene involved in T and B cell proliferation and survival through AKT pathway enhancement. Promotes cell cycle progression and inhibits apoptosis. Overexpressed in certain lymphoid malignancies including chronic lymphocytic leukemia.",
        "clinical_significance": "While oncogenic in hematological malignancies, expression in breast cancer microenvironment may indicate specific lymphocyte populations. Can serve as marker of particular B cell and T cell subsets within tumors.",
        "pearson_correlation": 0.66
    },
    "TRAC": {
        "full_name": "T-Cell Receptor Alpha Constant Region",
        "function": "Constant region of T-cell receptor alpha chain, essential component of TCR complex paired with variable regions for antigen recognition. All mature αβ T cells express TRAC as part of their TCR.",
        "clinical_significance": "Marker of mature T lymphocytes in tumor microenvironment. Indicates presence of adaptive immune infiltration with antigen-specific T cells. High expression correlates with better prognosis and immunotherapy response.",
        "pearson_correlation": 0.88
    },
    
    # Immune-Related Cell Adhesion
    "CEACAM6": {
        "full_name": "Carcinoembryonic Antigen-Related Cell Adhesion Molecule 6 (CD66c)",
        "function": "GPI-anchored membrane protein involved in cell adhesion, invasion, and immune evasion. Activates PI3K/AKT and MAPK signaling pathways and promotes epithelial-mesenchymal transition.",
        "clinical_significance": "Overexpressed in multiple cancer types. High expression correlates with chemoresistance, immunosuppression through immune checkpoint upregulation, and poor survival in hormone receptor-positive breast cancer. Emerging as potential therapeutic target with antibody-drug conjugates and CAR-T approaches under investigation.",
        "pearson_correlation": 0.78
    },
    "CEACAM8": {
        "full_name": "Carcinoembryonic Antigen-Related Cell Adhesion Molecule 8 (CD66b)",
        "function": "Member of CEACAM family expressed primarily on granulocytes, particularly neutrophils. GPI-anchored cell surface glycoprotein involved in cell adhesion, neutrophil activation, and immune cell function.",
        "clinical_significance": "Marker of neutrophil infiltration in tumor microenvironment. May indicate inflammatory response or neutrophil extracellular trap (NET) formation. Tumor-associated neutrophils can have both pro- and anti-tumor effects depending on polarization state (N1 vs N2 phenotype).",
        "pearson_correlation": 0.59
    },
    
    # Lipid Metabolism
    "FASN": {
        "full_name": "Fatty Acid Synthase",
        "function": "Rate-limiting enzyme in de novo fatty acid synthesis pathway. Catalyzes synthesis of palmitic acid from acetyl-CoA and malonyl-CoA using NADPH as reducing agent. Multi-enzyme complex with 7 catalytic domains performing all steps of fatty acid biosynthesis.",
        "clinical_significance": "Dramatically upregulated in breast cancer to meet increased lipid demands for membrane synthesis, signaling molecules, and energy storage. Associated with poor prognosis, therapeutic resistance, and more aggressive tumor phenotype. Promising metabolic therapy target with several inhibitors in development including TVB-2640.",
        "pearson_correlation": 0.89
    },
    "SCD": {
        "full_name": "Stearoyl-CoA Desaturase (Delta-9-Desaturase)",
        "function": "Rate-limiting enzyme converting saturated fatty acids (stearic acid, palmitic acid) to monounsaturated fatty acids (oleic acid, palmitoleic acid) by introducing double bond at delta-9 position. Critical for regulating membrane fluidity, lipid metabolism, and cellular signaling.",
        "clinical_significance": "Upregulated in cancer cells to support proliferation, survival, and chemoresistance. Inhibition induces ER stress, lipotoxicity, and apoptosis in cancer cells while sparing normal cells. Emerging as potential therapeutic target with inhibitors showing preclinical efficacy.",
        "pearson_correlation": 0.84
    },
    "ADIPOQ": {
        "full_name": "Adiponectin",
        "function": "Adipocyte-secreted hormone (adipokine) with anti-inflammatory, insulin-sensitizing, and anti-angiogenic properties. Activates AMPK and PPARα pathways and modulates glucose and lipid metabolism.",
        "clinical_significance": "Low adiponectin levels (hypoadiponectinemia) associated with increased breast cancer risk, particularly in obese and postmenopausal women. Higher levels correlate with better prognosis and reduced recurrence risk. Represents the obesity-cancer connection as obesity causes decreased adiponectin and increased leptin.",
        "pearson_correlation": 0.49
    },
    
    # Water & Small Molecule Transport
    "AQP1": {
        "full_name": "Aquaporin 1 (CHIP28)",
        "function": "Water channel protein expressed in endothelial cells, erythrocytes, renal tubules, and some epithelial tissues. Facilitates rapid bidirectional water transport across cell membranes (up to 3 billion water molecules per second per channel).",
        "clinical_significance": "Overexpression in breast cancer correlates with angiogenesis, increased cell migration, and metastatic potential. Associated with tumor aggressiveness, poor prognosis, and reduced survival. May facilitate rapid cell volume changes during migration through confined spaces.",
        "pearson_correlation": 0.72
    },
    "AQP3": {
        "full_name": "Aquaporin 3",
        "function": "Aquaglyceroporin that transports water, glycerol, urea, and small neutral solutes. Expressed in basolateral membranes of epithelial cells in skin, airways, and other tissues. Facilitates glycerol transport for lipid metabolism and cell proliferation.",
        "clinical_significance": "Upregulated in breast cancer where it promotes cell proliferation, migration, and invasion. Contributes to glycerol metabolism supporting cancer cell energetics and biosynthesis. AQP3 expression correlates with worse prognosis. May be therapeutic target with small molecule inhibitors showing preclinical efficacy.",
        "pearson_correlation": 0.90
    },
    
    # Secreted & Glandular Markers
    "PIGR": {
        "full_name": "Polymeric Immunoglobulin Receptor",
        "function": "Transports polymeric immunoglobulins (IgA dimers and pentameric IgM) across epithelial barriers from basolateral to apical surface. Critical for mucosal immunity and secretory antibody response.",
        "clinical_significance": "Expression indicates secretory epithelial phenotype and functional immune interactions at epithelial surfaces. May reflect preserved epithelial differentiation. PIGR expression patterns vary across breast cancer subtypes and may influence local immune responses.",
        "pearson_correlation": 0.53
    },
    "MUC6": {
        "full_name": "Mucin 6 (Gastric Mucin)",
        "function": "Gel-forming mucin primarily expressed in gastric epithelium and some breast tissues. Oligomerizes to form viscous gel providing protective barrier function against pathogens, acid, and mechanical stress.",
        "clinical_significance": "Aberrant expression may indicate glandular differentiation patterns in breast tumors. Associated with specific histological subtypes including mucinous carcinoma. May reflect differentiation status of tumor cells.",
        "pearson_correlation": 0.77
    },
    "SCGB2A1": {
        "full_name": "Secretoglobin Family 2A Member 1 (Mammaglobin-A)",
        "function": "Small secreted protein (10 kDa) highly specific to breast and prostate tissues. Belongs to secretoglobin superfamily with potential anti-inflammatory and immunomodulatory properties.",
        "clinical_significance": "Expressed in majority (70-80%) of breast cancers regardless of subtype. Excellent marker for detecting circulating tumor cells (CTCs), bone marrow micrometastases, and sentinel lymph node involvement from breast cancer primary. Used in RT-PCR assays for minimal residual disease detection.",
        "pearson_correlation": 0.64
    },
    "PTGDS": {
        "full_name": "Prostaglandin D2 Synthase (Lipocalin-Type)",
        "function": "Catalyzes production of prostaglandin D2 (PGD2) from prostaglandin H2. Involved in inflammation resolution, sleep regulation, neuroprotection, and potentially tumor suppression. Acts as lipocalin transporter for small lipophilic molecules.",
        "clinical_significance": "May have protective anti-inflammatory and anti-proliferative effects. Expression patterns vary across breast cancer subtypes with higher expression in some favorable prognosis subtypes. PGD2 signaling may inhibit tumor growth and angiogenesis through DP1 and DP2 receptors.",
        "pearson_correlation": 0.68
    },
    "TPSAB1": {
        "full_name": "Tryptase Alpha/Beta 1",
        "function": "Serine protease stored in mast cell secretory granules at extraordinarily high concentrations. Released during mast cell degranulation in allergic and inflammatory responses. Cleaves various substrates including complement, fibrinogen, and protease-activated receptors.",
        "clinical_significance": "Marker of mast cell infiltration in tumor microenvironment. Elevated serum tryptase indicates systemic mast cell activation. Mast cells can have both protumorigenic effects (angiogenesis, immunosuppression, matrix remodeling) and antitumorigenic effects (cytotoxicity, immune activation) depending on activation state and context.",
        "pearson_correlation": 0.23
    },
    "SERPINA3": {
        "full_name": "Serpin Family A Member 3 (Alpha-1-Antichymotrypsin)",
        "function": "Serine protease inhibitor that regulates proteolytic cascades and inflammatory responses. Acute phase reactant upregulated during inflammation, infection, and tissue damage. Inhibits neutrophil cathepsin G and mast cell chymase.",
        "clinical_significance": "Can be upregulated in cancer and may modulate tumor microenvironment by regulating protease activity. May indicate inflammatory response or tissue remodeling. Expression correlates with advanced stage in some cancers. Can have dual roles in promoting or inhibiting tumor progression.",
        "pearson_correlation": 0.87
    },
    "PTN": {
        "full_name": "Pleiotrophin (Heparin-Binding Growth Factor 8)",
        "function": "Heparin-binding growth factor involved in angiogenesis, cell migration, tissue remodeling, and neurite outgrowth. Binds to cell surface heparan sulfate proteoglycans, receptor protein tyrosine phosphatase β/ζ, and integrins.",
        "clinical_significance": "Overexpression promotes tumor growth, angiogenesis, metastasis, and chemoresistance. Associated with poor prognosis in multiple cancer types including breast cancer. Secreted into tumor microenvironment where it acts as paracrine factor. Potential therapeutic target with inhibitors under development.",
        "pearson_correlation": 0.54
    },
    
    # Cytoskeletal & Contractile Proteins
    "MYH11": {
        "full_name": "Myosin Heavy Chain 11 (Smooth Muscle Myosin)",
        "function": "Smooth muscle-specific myosin heavy chain essential for smooth muscle contraction. Major component of myoepithelial cells surrounding mammary ducts and alveoli. Forms thick filaments that interact with actin for force generation.",
        "clinical_significance": "Marker of myoepithelial cells in breast tissue used in immunohistochemistry. Loss or absence of myoepithelial layer (indicated by absent MYH11, p63, or calponin staining) distinguishes invasive carcinoma from ductal carcinoma in situ (DCIS). Intact myoepithelial layer contains in situ disease; breached layer allows invasion.",
        "pearson_correlation": 0.52
    },
    "MYLK": {
        "full_name": "Myosin Light Chain Kinase",
        "function": "Calcium/calmodulin-dependent enzyme that phosphorylates myosin regulatory light chains (MLC) to initiate smooth muscle contraction and regulate non-muscle cell motility. Key regulator of actomyosin contractility and cytoskeletal dynamics.",
        "clinical_significance": "Involved in cytoskeletal remodeling, cell migration, invasion, and metastasis in cancer cells. Promotes stress fiber formation and cell contractility required for migration through extracellular matrix. MYLK activation drives tumor cell invasion and may be therapeutic target. Overexpression associated with metastatic phenotype.",
        "pearson_correlation": 0.64
    },
    "MYBPC1": {
        "full_name": "Myosin Binding Protein C1 (Slow Skeletal Muscle)",
        "function": "Structural protein that regulates skeletal muscle contraction by modulating myosin-actin interactions. Predominantly expressed in slow-twitch (type I) muscle fibers. Binds to myosin heavy chain and titin in sarcomere.",
        "clinical_significance": "May indicate muscle or myoepithelial differentiation patterns in tumors. Expression changes may reflect altered contractile properties. Aberrant expression in non-muscle tissues like breast tumors suggests differentiation abnormalities.",
        "pearson_correlation": 0.90
    },
    
    # Signaling & Regulatory Proteins
    "SFRP1": {
        "full_name": "Secreted Frizzled-Related Protein 1",
        "function": "Soluble antagonist of Wnt signaling pathway that competes with Wnt ligands for frizzled receptor binding. Contains cysteine-rich domain homologous to frizzled receptors. Functions as tumor suppressor by inhibiting Wnt-mediated proliferation, invasion, and stem cell properties.",
        "clinical_significance": "Frequently silenced by promoter hypermethylation in breast cancer (>80% of cases). Loss promotes tumor growth, invasion, metastasis, and cancer stem cell expansion. Methylation status is potential biomarker for prognosis and treatment response. Re-expression or Wnt pathway inhibition are therapeutic strategies.",
        "pearson_correlation": 0.75
    },
    "CLIC6": {
        "full_name": "Chloride Intracellular Channel 6",
        "function": "Member of chloride intracellular channel family involved in intracellular chloride transport, pH regulation, cell volume control, and apoptosis. Localized to intracellular membranes and can insert into lipid bilayers forming ion channels.",
        "clinical_significance": "Emerging role in breast cancer with potential involvement in cell cycle regulation, apoptosis resistance, and cellular stress responses. May be dysregulated in cancer cells. Function in cancer requires further investigation but may regulate organellar pH and membrane potential.",
        "pearson_correlation": 0.93
    },
    "CYTIP": {
        "full_name": "Cytohesin 1 Interacting Protein (PSCDBP)",
        "function": "Regulates ARF-dependent signaling pathways and membrane trafficking. Interacts with cytohesin family guanine nucleotide exchange factors affecting small GTPase signaling, particularly ARF family proteins. Involved in vesicle trafficking and cell adhesion.",
        "clinical_significance": "May modulate cell adhesion, migration pathways, and intracellular trafficking processes relevant to cancer progression. Regulates β-integrin activation and cell-matrix adhesion. Potential role in controlling cell motility and invasion.",
        "pearson_correlation": 0.75
    },
    "OPRPN": {
        "full_name": "Opiorphin (Putative)",
        "function": "Limited information available. May be involved in pain modulation, inflammation, or other regulatory processes. Opiorphin is a natural peptide with analgesic properties.",
        "clinical_significance": "Role in breast cancer biology requires further investigation. May participate in pain-related or inflammatory pathways in the tumor microenvironment.",
        "pearson_correlation": 0.27
    },
    "SERHL2": {
        "full_name": "Serine Hydrolase-Like 2",
        "function": "Putative serine hydrolase with incompletely characterized enzymatic activity and substrate specificity. May be involved in lipid metabolism or other hydrolytic processes. Belongs to α/β-hydrolase fold superfamily.",
        "clinical_significance": "Function in breast cancer biology requires further investigation. Potential metabolic enzyme with unclear clinical relevance. May participate in lipid signaling or metabolism pathways.",
        "pearson_correlation": 0.87
    },
    "TENT5C": {
        "full_name": "Terminal Nucleotidyltransferase 5C (FAM46C)",
        "function": "RNA-modifying enzyme that adds non-templated nucleotides to RNA molecules (particularly poly(A) tails). Involved in RNA polyadenylation, stability regulation, and mRNA degradation. Member of nucleotidyltransferase family.",
        "clinical_significance": "Emerging role in post-transcriptional gene regulation. May influence mRNA stability, translation efficiency, and gene expression in cancer cells. FAM46C (in same family) is tumor suppressor frequently mutated in multiple myeloma. TENT5C function in breast cancer requires further study.",
        "pearson_correlation": 0.91
    },
    "ANKRD30A": {
        "full_name": "Ankyrin Repeat Domain 30A",
        "function": "Contains multiple ankyrin repeats which mediate protein-protein interactions. Specific cellular functions not fully characterized. Ankyrin repeats are common motifs found in diverse proteins involved in cell signaling, transcription, and structural organization.",
        "clinical_significance": "May participate in cellular signaling networks or structural organization. Role in breast cancer requires further study. Some evidence suggests involvement in cancer-related pathways but mechanism unclear.",
        "pearson_correlation": 0.93
    },
    "ABCC11": {
        "full_name": "ATP Binding Cassette Subfamily C Member 11 (MRP8)",
        "function": "ATP-dependent transporter involved in efflux of various endogenous compounds including cyclic nucleotides and conjugated steroids. Member of multidrug resistance-associated protein family. SNP (538G>A) determines earwax type (wet vs dry) and axillary odor phenotype in East Asian populations.",
        "clinical_significance": "May influence chemotherapy drug efflux and contribute to treatment resistance. Genetic variants affect drug metabolism, toxicity profiles, and body odor/earwax phenotype. Expression in breast tissue may impact drug distribution and therapeutic response. Potential role in endocrine therapy resistance through steroid conjugate transport.",
        "pearson_correlation": 0.87
    },
    
    # Structural & Adhesion Proteins
    "DST": {
        "full_name": "Dystonin (BPAG1, Bullous Pemphigoid Antigen 1)",
        "function": "Extremely large cytoskeletal linker protein (>600 kDa) that connects intermediate filaments (cytokeratins) to microtubules and actin filaments. Essential for cytoskeletal organization, cellular mechanical stability, and organelle positioning.",
        "clinical_significance": "Loss may contribute to cytoskeletal disorganization in cancer cells, affecting cell shape, polarity, motility, and mechanical properties. Disruption can impair cellular mechanical integrity and facilitate metastatic behavior. Mutations cause epidermolysis bullosa and hereditary sensory neuropathy.",
        "pearson_correlation": 0.82
    },
    "VWF": {
        "full_name": "Von Willebrand Factor",
        "function": "Large multimeric glycoprotein (up to 20,000 kDa multimers) essential for hemostasis and platelet adhesion to damaged endothelium. Synthesized by endothelial cells and megakaryocytes. Stored in Weibel-Palade bodies and released upon endothelial activation.",
        "clinical_significance": "Marker of endothelial cells and tumor vasculature used in immunohistochemistry. Expression indicates angiogenic activity and vascular density in tumors. High microvessel density (MVD) by VWF or CD31 staining correlates with metastatic potential and poor prognosis. Can assess tumor angiogenesis and response to anti-angiogenic therapies.",
        "pearson_correlation": 0.52
    },
    "TACSTD2": {
        "full_name": "Tumor-Associated Calcium Signal Transducer 2 (TROP2, EGP-1)",
        "function": "Transmembrane cell surface glycoprotein involved in calcium-dependent cell adhesion and intracellular signaling. Regulates cell growth, differentiation, and proliferation. Intracellular domain cleaved and translocates to nucleus acting as transcriptional regulator.",
        "clinical_significance": "Overexpressed in multiple epithelial cancers including breast cancer (>90% of breast cancers). FDA-approved antibody-drug conjugate sacituzumab govitecan (Trodelvy) targets TROP2 in metastatic triple-negative and HR+/HER2- breast cancer showing significant survival benefit. Major therapeutic target with multiple TROP2-directed therapies in development.",
        "pearson_correlation": 0.88
    },
    
    # Enzyme & Metabolic Function
    "ADH1B": {
        "full_name": "Alcohol Dehydrogenase 1B (Beta Subunit)",
        "function": "Primary enzyme for ethanol oxidation to acetaldehyde in liver and other tissues. Class I alcohol dehydrogenase composed of alpha and beta subunits. Uses NAD+ as cofactor and metabolizes various alcohols. Polymorphisms significantly affect enzyme kinetics and alcohol metabolism rate.",
        "clinical_significance": "Genetic variants influence alcohol metabolism efficiency, intoxication susceptibility, and alcohol-related disease risk including cancer. ADH1B*2 and *3 alleles encode fast-metabolizing enzymes causing rapid acetaldehyde accumulation. Slow metabolizers may have different breast cancer risk profiles related to alcohol exposure. Alcohol is established risk factor for breast cancer (7% increased risk per 10g/day).",
        "pearson_correlation": 0.59
    }
}

# Get category for a gene
def get_gene_category(gene_symbol):
    for category, genes in GENE_CATEGORIES.items():
        if gene_symbol in genes:
            return category
    return "Unknown"

# Get all genes in a category
def get_genes_in_category(category):
    return GENE_CATEGORIES.get(category, [])

# Get gene info
def get_gene_info(gene_symbol):
    return GENE_INFO.get(gene_symbol, {
        "full_name": gene_symbol,
        "function": "Information not available",
        "clinical_significance": "Information not available",
        "pearson_correlation": 0.57
    })


