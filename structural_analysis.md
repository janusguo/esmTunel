# Conformational Epitope Analysis and Mapping Report

## 1. Overview
Conformational (structural) epitopes are critical for understanding allergenicity. Unlike linear epitopes, they depend on the 3D folding of the protein. This report identifies potential structural epitopes in our current dataset and proposes a method to leverage ESM-2/3 models to highlight these features.

## 2. Key Allergen Families and Structural Epitopes

### 2.1 Bet v 1 Homologs (e.g., Mal d 1, Ara h 8)
*   **Representative ID**: `sp|P43211|MAL11_MALDO` (Mal d 1)
*   **Sequence**: `MGVYTFENEFTSEIPPSRLFKAFVLDADNLIPKIAPQAIKQAEILEGNGGPGTIKKITFGEGSQYGYVKHRIDSIDEASYSYSYTLIEGDALTDTIEKISYETKLVACGSGSTIKSISHYHTKGNIEIKEEHVKVGKEKAHGLFKLIESYLKDHPDAYN`
*   **Structure**: A 7-stranded antiparallel β-sheet wrapping around a large C-terminal α-helix.
*   **Potential Conformational Epitopes**:
    *   **P-loop region**: Residues 46-52 (`GNGGPGT`). This loop is highly surface-exposed and a known target for IgE in Bet v 1 homologs.
    *   **C-terminal Helix**: Residues 146-159 (`LIESYLKDHPDAYN`).
    *   **Cross-reactivity**: The structural similarity between Bet v 1 and Mal d 1 explains the "Birch-Apple Syndrome".

### 2.2 Lipid Transfer Proteins (LTPs)
*   **Representative ID**: `sp|Q9M5X7|NLTP_MALDO` (Mal d 3)
*   **Structure**: Compact bundle of four α-helices stabilized by four disulfide bridges.
*   **Potential Conformational Epitopes**:
    *   Surface loops connecting the helices.
    *   Residues near the hydrophobic cavity.
    *   Stability: The structural rigidity (due to disulfide bonds) allows these epitopes to remain intact even after thermal processing or digestion.

### 2.3 Profilins
*   **Representative ID**: `sp|P85984|PROF_BETVU`
*   **Structure**: Central β-sheet flanked by α-helices.
*   **Potential Conformational Epitopes**:
    *   Profilins show high structural conservation (>70% identity), leading to broad IgE cross-reactivity.
    *   The epitope regions are typically spread across the surface-exposed β-strands.

## 3. Mapping Methodologies

### 3.1 Known Structural Information (PDB)
If structural data is available, we can map epitopes from IEDB:
-   **Bet v 1**: PDB 1BTV, 4AWH. Epitopes mapped to surface-exposed patches.
-   **Mal d 1**: PDB 5KVY.
-   **Phl p 1**: PDB 1N10.

### 3.2 ESM-2 Hidden State Analysis
ESM models capture structural information in their attention heads and hidden states.
*   **Attention Maps**: High attention between residues $i$ and $j$ where $|i-j| > 10$ indicates potential 3D proximity. Clusters of such contacts on the protein surface are likely candidates for conformational epitopes.
*   **Saliency Mapping**: By calculating the gradient of the allergenicity score with respect to the input embeddings, we can identify "important" residues. If these residues are spatially proximal in 3D models (like AlphaFold), they form a structural epitope.

## 4. Recommendations for ML Integration
1.  **Extract Contacts**: Use the ESM-2 contact prediction head to generate a contact map for all sequences in `test.csv`.
2.  **Filter for Allergens**: For high-confidence allergen predictions, visualize the attention maps for layers 30-33 (in the 650M model), which are known to be structure-rich.
3.  **Cross-validation**: Compare predicted "important" residues with known epitope regions for Bet v 1 and LTPs.

## 5. Conclusion
Conformational epitopes are the "smoking gun" of allergenicity. By analyzing non-local attention in ESM, we can move beyond sequence-level binary classification and provide interpretability regarding *why* a protein is an allergen (i.e., which structural features are recognized by the immune system).
