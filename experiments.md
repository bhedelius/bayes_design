# Experiments

## Designed proteins

### WN Model System
Original Sequence:
```
KLPPGWEKRMSRSSGRVYYFNHITNASQFERPSG
```
Original sequence with mask, preserving fixed residues:
```
---------------------------------G
```
Sequence Stats:
- BayesDesign LogP: -89.15804301725373 `bayes-design-experiment compare_seq_probs --protein_id 1PIN --model_name bayes_design --decode_order n_to_c --fixed_positions 34 34 --sequences KLPPGWEKRMSRSSGRVYYFNHITNASQFERPSG`
- ProteinMPNN LogP: -64.23751689474135 `bayes-design-experiment compare_seq_probs --protein_id 1PIN --model_name protein_mpnn --decode_order n_to_c --fixed_positions 34 34 --sequences KLPPGWEKRMSRSSGRVYYFNHITNASQFERPSG`

#### BayesDesign
Command:
```
bayes-design --model_name bayes_design --protein_id 1PIN --decode_order n_to_c --decode_algorithm beam_medium --n_beams 128 --fixed_positions 34 34
```
Sequence:
```
MLPEGWVKQRNPITGEDVCFNTLTHEMTKFEPQG
```
Sequence Stats:
- BayesDesign LogP: -31.573002420457204 `bayes-design-experiment compare_seq_probs --protein_id 1PIN --model_name bayes_design --decode_order n_to_c --fixed_positions 34 34 --sequences MLPEGWVKQRNPITGEDVCFNTLTHEMTKFEPQG`
- ProteinMPNN LogP: -52.71265054032396 `bayes-design-experiment compare_seq_probs --protein_id 1PIN --model_name protein_mpnn --decode_order n_to_c --fixed_positions 34 34 --sequences MLPEGWVKQRNPITGEDVCFNTLTHEMTKFEPQG`


#### ProteinMPNN
Command:
```
bayes-design --model_name protein_mpnn --protein_id 1PIN --decode_order n_to_c --decode_algorithm beam_medium --n_beams 128 --fixed_positions 34 34
```
Sequence:
```
TLPEGWVERVDPKTGEKVFFNTKTKEVTKEKPVG
```
Sequence Stats:
- BayesDesign LogP: -73.07880518446302 `bayes-design-experiment compare_seq_probs --protein_id 1PIN --model_name bayes_design --decode_order n_to_c --fixed_positions 34 34 --sequences TLPEGWVERVDPKTGEKVFFNTKTKEVTKEKPVG`
- ProteinMPNN LogP: -30.726229193957487 `bayes-design-experiment compare_seq_probs --protein_id 1PIN --model_name protein_mpnn --decode_order n_to_c --fixed_positions 34 34 --sequences TLPEGWVERVDPKTGEKVFFNTKTKEVTKEKPVG`

### Plastic Degrading Enzyme
Original Sequence:
```
TDPGNGSGYQRGPDPTVSFLEAARGQYTVDTERVSSLVGGFGGGTIHYPEDVSGTMAAIVVIPGYVSAESSIEWWGPKLASYGFVVMTIDTNTGFDQPPSRATQINAALDYLVDQNSDNGSPVQGMIDTSRLGVIGWSMGGGGTIRVASQGRIKAAIPLAPWDTSSYYARRAEAATMIFACESDVVAPVGLHASPFYNALPSSIDKAFVEINNGSHFCANGGGINNDVLGRLGVSWMKLHLDEDGRYNQFLCGPNHESDFSISEYRGNCPYGSHHHHHHHH
```
Sequence Stats:
- BayesDesign LogP:
- ProteinMPNN LogP: 

Original sequence with mask, preserving fixed residues:
```
----------------------------------------------------------------YVS-----------------------------Q---------------------------------------WSM----------------------W------------------C--D--------------------------IN--SHFC--GGGINN-------------------------C--------S-------C----HHHHHHHH
```

#### BayesDesign
Command:
```
bayes-design --model_name bayes_design --protein_id plastic_degrading_enzyme --decode_order n_to_c --decode_algorithm beam_medium --n_beams 128 --fixed_positions 138 138 184 184 216 216 65 65 97 97 139 139 162 162 66 67 137 137 211 212 215 215 217 217 223 225 261 261 221 226 181 181 218 218 252 252 269 269 274 281
```
Sequence:
```
AIQWGKDAHVRGPIPTKELLCADHGFWEVSTQDIPASVKGFGGWTIHFPQNWEGKLPAVVCMPGYVSNADSIAWWGPRLASFGFVVAVINHTRPDCQPEEMAQEIKAAMDHLIAQNKDPQSPIHGKIDENRLGVIGWSMGGGATIIVASDGRVKAAIPLCPWLNSTEPAKKATADVLIFSCENDTVCPPEKHSIPMWEAIPKEIDRMHVMINNGSHFCATGGGINNCTLNLLAVSWLRLHLMNDQSVEKFLCGPEIDNDPSISKKESNCPFGQHHHHHHHH
```
Sequence Stats:
- BayesDesign LogP: -212.90735622198204
- ProteinMPNN LogP: 

#### ProteinMPNN
Command:
```
bayes-design --model_name protein_mpnn --protein_id plastic_degrading_enzyme --decode_order n_to_c --decode_algorithm beam_medium --n_beams 128 --fixed_positions 138 138 184 184 216 216 65 65 97 97 139 139 162 162 66 67 137 137 211 212 215 215 217 217 223 225 261 261 221 226 181 181 218 218 252 252 269 269 274 281
```
Sequence:
```
AVAPGGHPLVRGPAPTAALLAAPAGPWAVATEAIPASVAGFGGGTVHFPLDVEGRLPAVVIAPGYVSSADSIAWWGPALASHGFVVLVINWTSPTVQPAEGAAEIEAALAHLDAANDDPASPIRGLIDRDRRGVIGWSMGGGATVIVASTGRVRAAIPLAPWLPSTEPAARATAATLIISCENDTVTPPEKFSIPIFEALPPEIDRMLVQINNGSHFCATGGGINNATLRRLVIAWLRLHLMDDPRVEAFLCGPAIESDPSISRFESNCPFGRHHHHHHHH
```
Sequence Stats:
- BayesDesign LogP:
- ProteinMPNN LogP: -180.78068155687816

### Recently Evolved Rice Protein (dn47)

Original Sequence:
```
STASSSLSHDALVDTITTAVADAIRISYQSPPGLTVDDAVTKSIADSAADKASAAVRDALSKDLPSPIAI  STASSSLSHDALVDTITTAVADAIRISYQSPPGLTVDDAVTKSIADSAADKASAAVRDALSKDLPSPIAI
```
Original Sequence with mask, preserving fixed residues:
```
--------------------------------------------------------------------------------------------------------------------------------------------
```
Sequence Stats:
- BayesDesign LogP: -944.9901678422955 `bayes-design-experiment compare_seq_probs --protein_id dn47 --model_name bayes_design --decode_order n_to_c --sequences STASSSLSHDALVDTITTAVADAIRISYQSPPGLTVDDAVTKSIADSAADKASAAVRDALSKDLPSPIAISTASSSLSHDALVDTITTAVADAIRISYQSPPGLTVDDAVTKSIADSAADKASAAVRDALSKDLPSPIAI`
- ProteinMPNN LogP: -268.70771463316356 `bayes-design-experiment compare_seq_probs --protein_id dn47 --model_name bayes_design --decode_order n_to_c --sequences STASSSLSHDALVDTITTAVADAIRISYQSPPGLTVDDAVTKSIADSAADKASAAVRDALSKDLPSPIAISTASSSLSHDALVDTITTAVADAIRISYQSPPGLTVDDAVTKSIADSAADKASAAVRDALSKDLPSPIAI`

Hypotheses:
- BayesDesign increases specificity
    - Redesign protein for which the WT takes multiple conformations
        - SA + JS Try to increase specificity of active enzyme conformation, test enzymatic activity
            - Kinase - stabilize active conformation
                - SA - Find kinase with active conformation, send to JS
                - JS - Redesign to prefer active conformation
        - Test alpha-beta conformation protein with CD
        - SAX to see % of redesigned protein in each conformation
    - Amyloid beta sheets

- By increasing specificity, we increase stability (especially if unfolded state is included in the energy landscape)
    - Redesign ligand protein (nanobody, HIV protein) to bind with more affinity
        - Measure k_d
    - AM + JS - CD at different temperatures or differential scanning fluorometry
        - PETase
        - dn_47
        - HIV protein (JS redesign)
    - AM - Measure change in enzyme activitity
        - PETase

- BayesDesign helps avoid ProteinMPNN repetition problem for de novo proteins
    - JG + JS - Redesign de novo rice proteins
    - JG + JS - Redesign random protein (g c bias)
        - Find other weird ProteinMPNN cases
        - Show that AlphaFold is also unexpectedly confident on those cases

- Unconstrained backbone helps us design even more specific structures
    - JS - work on unconstrained backbone case
    
    

maximum p(seq|struct) / p(seq) = maximum log p(seq|struct) - log p(seq)

#### BayesDesign
Command:
```
bayes-design --model_name bayes_design --protein_id dn47 --decode_order n_to_c --decode_algorithm beam_medium --n_beams 128
```
Sequence:
```
LNEHMAISVESLKQKWCEAHEQAVQDAFTRPEGWDIPEQVWQARANKAACLSCQKVNEVMDKLIPPESERKFKNDPLTVDKLCDACCQAVCWAVQEALSEPPGVDWPREQQDHIAQKSAQKTCEGTRAVMNECIEPPKIM
```
Sequence Stats:
- BayesDesign LogP: -163.4057692585695 `bayes-design-experiment compare_seq_probs --protein_id dn47 --model_name bayes_design --decode_order n_to_c --sequences LNEHMAISVESLKQKWCEAHEQAVQDAFTRPEGWDIPEQVWQARANKAACLSCQKVNEVMDKLIPPESERKFKNDPLTVDKLCDACCQAVCWAVQEALSEPPGVDWPREQQDHIAQKSAQKTCEGTRAVMNECIEPPKIM`
- ProteinMPNN LogP: -165.74089450285308 `bayes-design-experiment compare_seq_probs --protein_id dn47 --model_name bayes_design --decode_order n_to_c --sequences LNEHMAISVESLKQKWCEAHEQAVQDAFTRPEGWDIPEQVWQARANKAACLSCQKVNEVMDKLIPPESERKFKNDPLTVDKLCDACCQAVCWAVQEALSEPPGVDWPREQQDHIAQKSAQKTCEGTRAVMNECIEPPKIM`


#### ProteinMPNN
Command:
```
bayes-design --model_name protein_mpnn --protein_id dn47 --decode_order n_to_c --decode_algorithm beam_medium --n_beams 128
```
Sequence:
```
PPPKKKPSKEEIIKKVVEAVKKAVKEAFKKPEGLDIPEEEWEKIAEKAAEKAAKKVEEVLKKEIPPEEEKKKKKKKLTKEKIIEKIAKAVKEAVKEALKEPEGLDIPEELAKKIAEEAAKKAKKEVEKVLKEVIPKKKKK
```
Sequence Stats:
- BayesDesign = ProteinMPNN / XLNet LogP: -578.6982698784458 `bayes-design-experiment compare_seq_probs --protein_id dn47 --model_name bayes_design --decode_order n_to_c --sequences PPPKKKPSKEEIIKKVVEAVKKAVKEAFKKPEGLDIPEEEWEKIAEKAAEKAAKKVEEVLKKEIPPEEEKKKKKKKLTKEKIIEKIAKAVKEAVKEALKEPEGLDIPEELAKKIAEEAAKKAKKEVEKVLKEVIPKKKKK`
- ProteinMPNN LogP: -101.24846088136692 `bayes-design-experiment compare_seq_probs --protein_id dn47 --model_name protein_mpnn --decode_order n_to_c --sequences PPPKKKPSKEEIIKKVVEAVKKAVKEAFKKPEGLDIPEEEWEKIAEKAAEKAAKKVEEVLKKEIPPEEEKKKKKKKLTKEKIIEKIAKAVKEAVKEALKEPEGLDIPEELAKKIAEEAAKKAKKEVEKVLKEVIPKKKKK`

### Recently Evolved Rice Protein (dn47_cut2)

Original Sequence:
```
STASSSLSHDALVDTITTAVADAIRISYQSPPGLTVDDAVTKSIADSAADKASAAVRDALSKDLPSPLSHDALVDTITTAVADAIRISYQSPPGLTVDDAVTKSIADSAADKASAAVRDALSKDLPSPIAI
```
Sequence Stats:
- BayesDesign LogP: -867.6463986468442 `bayes-design-experiment compare_seq_probs --protein_id dn47_cut2 --model_name bayes_design --decode_order n_to_c --sequences STASSSLSHDALVDTITTAVADAIRISYQSPPGLTVDDAVTKSIADSAADKASAAVRDALSKDLPSPLSHDALVDTITTAVADAIRISYQSPPGLTVDDAVTKSIADSAADKASAAVRDALSKDLPSPIAI`
- ProteinMPNN LogP: -245.032467840132 `bayes-design-experiment compare_seq_probs --protein_id dn47_cut2 --model_name protein_mpnn --decode_order n_to_c --sequences STASSSLSHDALVDTITTAVADAIRISYQSPPGLTVDDAVTKSIADSAADKASAAVRDALSKDLPSPLSHDALVDTITTAVADAIRISYQSPPGLTVDDAVTKSIADSAADKASAAVRDALSKDLPSPIAI`

#### BayesDesign
Command:
```
bayes-design --model_name bayes_design --protein_id dn47_cut2 --decode_order n_to_c --decode_algorithm beam_medium --n_beams 128
```
Sequence:
```
ECSMHMCTEDCFIDQVVLAVSQAIKQWFEPESGQAIPREQINAHADKCAQMGSQAVRDVLKEKLPCNVSPEEFIELVVKAHSQACLQALTMPEGQAWPEFQRNEHANGAAKGVSQACKCVIDECIPKPPDI
```
Sequence Stats:
- BayesDesign LogP: -142.98842013604815 `bayes-design-experiment compare_seq_probs --protein_id dn47_cut2 --model_name bayes_design --decode_order n_to_c --sequences ECSMHMCTEDCFIDQVVLAVSQAIKQWFEPESGQAIPREQINAHADKCAQMGSQAVRDVLKEKLPCNVSPEEFIELVVKAHSQACLQALTMPEGQAWPEFQRNEHANGAAKGVSQACKCVIDECIPKPPDI`
- ProteinMPNN LogP: -222.6793760330641 `bayes-design-experiment compare_seq_probs --protein_id dn47_cut2 --model_name protein_mpnn --decode_order n_to_c --sequences ECSMHMCTEDCFIDQVVLAVSQAIKQWFEPESGQAIPREQINAHADKCAQMGSQAVRDVLKEKLPCNVSPEEFIELVVKAHSQACLQALTMPEGQAWPEFQRNEHANGAAKGVSQACKCVIDECIPKPPDI`


#### ProteinMPNN
Command:
```
bayes-design --model_name protein_mpnn --protein_id dn47_cut2 --decode_order n_to_c --decode_algorithm beam_medium --n_beams 128
```
Sequence:
```
PPPPAAPTPEELREAVARAVREAVREVFRPPPGDPTPPELWERLAEEAARAAEEAVRRVLEERLPEPLPPEELREAIARAVREAVRRVLEGEPGDPIPEELRERLAEEAARAAREAVEEVLRRLLPPPPPA
```
Sequence Stats:
- BayesDesign LogP: -501.2846077182429 `bayes-design-experiment compare_seq_probs --protein_id dn47_cut2 --model_name bayes_design --decode_order n_to_c --sequences PPPPAAPTPEELREAVARAVREAVREVFRPPPGDPTPPELWERLAEEAARAAEEAVRRVLEERLPEPLPPEELREAIARAVREAVRRVLEGEPGDPIPEELRERLAEEAARAAREAVEEVLRRLLPPPPPA`
- ProteinMPNN LogP: -91.01211402479034 `bayes-design-experiment compare_seq_probs --protein_id dn47_cut2 --model_name protein_mpnn --decode_order n_to_c --sequences PPPPAAPTPEELREAVARAVREAVREVFRPPPGDPTPPELWERLAEEAARAAEEAVRRVLEERLPEPLPPEELREAIARAVREAVRRVLEGEPGDPIPEELRERLAEEAARAAREAVEEVLRRLLPPPPPA`

## Other experiments

### Conformational-specificity design (fold_specificity)
Design a sequence that prefers a target fold (`--protein_id`) over a competing
fold (`--decoy_id`), maximizing `p(seq | target) / p(seq | decoy)`. The two
backbones may be different (homologous) structures — the CLI aligns and crops
them to a common length, so `--fixed_positions` are 1-indexed positions in that
aligned sequence. This replaces the old `cs_design` model (`--protein_id_anti`).

Design the ERK kinase (4GSB) fold against the alternate 2ERK conformation,
keeping the termini fixed:
```
bayes-design --model_name fold_specificity --protein_id 4GSB --decoy_id 2ERK --decode_order n_to_c --decode_algorithm greedy --fixed_positions 1 168 187 358
```
Add `--ball_mask` to free only the residues within a ball around each designed
position (instead of fixing exactly `--fixed_positions`):
```
bayes-design --model_name fold_specificity --protein_id 4GSB --decoy_id 2ERK --decode_order n_to_c --decode_algorithm greedy --fixed_positions 1 168 187 358 --ball_mask
```
The reverse design (2ERK preferred over 4GSB) just swaps the two ids:
```
bayes-design --model_name fold_specificity --protein_id 2ERK --decoy_id 4GSB --decode_order n_to_c --decode_algorithm greedy --fixed_positions 1 168 187 358
```

### Backbone-ensemble objectives (theory-review Fix 1 & Fix 3)
`bayes_design_basin` averages the likelihood over a noise-perturbed ensemble of the
native backbone (conformational entropy). `bayes_design_selfconsistent` replaces the
XLNet marginal with a Monte Carlo partition function from the same ProteinMPNN over
that ensemble, with `--contrast_weight` tuning the denominator (the review's lambda).
`--n_ensemble` and `--ensemble_sigma` (angstroms) set the ensemble.
```
bayes-design --model_name bayes_design_basin --protein_id 6MRR --decode_algorithm greedy --fixed_positions 1 60 --n_ensemble 8 --ensemble_sigma 0.1
```
```
bayes-design --model_name bayes_design_selfconsistent --protein_id 6MRR --decode_algorithm greedy --fixed_positions 1 60 --n_ensemble 8 --ensemble_sigma 0.1 --contrast_weight 1.0
```
`bayes_design_decoy` is the self-consistent denominator with a genuinely diverse
background: the partition function runs over a library of different folds
(decoys / AlphaFold models) cropped to the design length, rather than noise around the
native backbone. `--decoy_library` points at the PDB directory and `--n_ensemble` caps
the decoys averaged. The library should exclude the target fold to avoid self-contamination.
```
bayes-design --model_name bayes_design_decoy --protein_id 6MRR --decode_algorithm greedy --fixed_positions 1 60 --decoy_library ./data/decoys --n_ensemble 16 --contrast_weight 1.0
```

### Calibrating Delta G against mega-scale stability data (Fix 2)
Fit the objective to measured folding stability (Tsuboyama et al. 2023), recovering the
denominator weight `lambda` and inverse temperature `beta` and comparing the XLNet
marginal against the self-consistent backbone-ensemble denominator. Needs the processed
dataset CSV and per-domain backbones (its AlphaFold PDBs); run on a GPU host. Start with
a small `--max-domains` to shake out structure/length coverage, then scale and reuse the
cached features:
```
python -m bayes_design.experiments.calibrate_dg --csv Tsuboyama2023_Dataset1_20230416.csv --structures-dir af_pdbs --schema dataset1 --max-domains 20 --features-out feat.csv --out calib.json
```
```
python -m bayes_design.experiments.calibrate_dg --csv Tsuboyama2023_Dataset1_20230416.csv --structures-dir af_pdbs --features-out feat.csv --reuse-features --out calib.json
```
Add `backbone_library` to `--denominators` to test the diverse-background denominator
(Fix 1) against the same data. `--decoy-library` is a directory of background backbones
(a pool distinct from the target domains, to avoid self-contamination) and `--n-decoys`
caps how many are averaged per variant:
```
python -m bayes_design.experiments.calibrate_dg --csv Tsuboyama2023_Dataset1_20230416.csv --structures-dir af_pdbs --denominators xlnet backbone_library --decoy-library decoy_pool --n-decoys 16 --max-domains 20 --features-out feat.csv --out calib.json
```

#### Result: XLNet beats the self-consistent backbone-ensemble denominator
Calibrated on 700 variants across 14 domains of Tsuboyama Dataset1, scoring each
variant on its wild-type AlphaFold backbone and differencing features within each
domain (target `ddG`). Correlations of predicted vs. measured `ddG`:

| Denominator | sigma | Pearson | Spearman | lambda | beta |
| --- | --- | --- | --- | --- | --- |
| none (numerator only) | - | 0.517 | 0.553 | - | - |
| backbone ensemble | 0.1 | 0.517 | 0.553 | 0.07 | 3.6 |
| backbone ensemble | 0.3 | 0.533 | 0.563 | -0.82 | 6.8 |
| backbone ensemble | 0.6 | 0.537 | 0.543 | -0.31 | 4.7 |
| backbone ensemble | 1.0 | 0.517 | 0.553 | -0.01 | 3.9 |
| XLNet p(seq) | - | 0.566 | 0.586 | -0.19 | 4.2 |

The XLNet marginal is the only denominator that improves ranking over the bare
numerator. The noise-perturbed self-consistent denominator does not: at small sigma
the differenced ensemble feature is collinear with the numerator, and at large sigma
the perturbed backbone stops discriminating mutations, so the feature collapses toward
zero. The erratic lambda signs across sigma confirm it is fitting noise, not negative
design. A self-consistent partition-function denominator needs a genuinely diverse
structural background (decoys / alternate folds), not Gaussian backbone noise.

Provenance: commit `d375dac`; Tsuboyama 2023 Dataset1
(`Processed_K50_dG_datasets/Tsuboyama2023_Dataset1_20230416.csv`, Zenodo 7844779) with
per-domain AlphaFold backbones; `bidirectional_mlm` scoring; run on ribo (Quadro RTX
5000); sigma sweep over a fixed 20-domain / 50-variant subset.

#### Result: the noise-ensemble numerator (Fix 3) does not recover conformational entropy
Fix 3 replaces the single-backbone likelihood `p(seq | b0)` with the basin average
`E_b[p(seq | b)]` (objective `bayes_design_basin`). Re-fitting the cached sigma-sweep
features (same 700 variants / 14 domains) with the ensemble as the numerator, correlations
of within-domain-differenced features against measured `ddG`:

| Numerator | sigma | Pearson | Spearman | corr(f_ens, f_num) |
| --- | --- | --- | --- | --- |
| single backbone (f_num) | - | 0.517 | 0.553 | - |
| noise ensemble (f_ens) | 0.1 | 0.507 | 0.537 | 0.98 |
| noise ensemble (f_ens) | 0.3 | 0.512 | 0.535 | 0.86 |
| noise ensemble (f_ens) | 0.6 | 0.384 | 0.288 | 0.50 |
| noise ensemble (f_ens) | 1.0 | 0.015 | 0.062 | 0.02 |

At the full-objective level (numerator plus XLNet denominator, sigma 0.1), `bayes_design`
(f_num + f_xlnet) gives 0.566 / 0.586 while `bayes_design_basin` (f_ens + f_xlnet) gives
0.560 / 0.576 -- slightly worse.

The noise ensemble never improves ranking. At small sigma it is nearly collinear with the
single backbone (0.98), so averaging only adds variance; as sigma grows the perturbed
backbones stop encoding the fold and the averaged likelihood collapses toward random
(sigma 1.0). This is the same failure, and the same cause, as the Fix 1 noise denominator:
isotropic Gaussian backbone noise is not a physical sample of the folded basin. Recovering
conformational entropy would need a physically meaningful backbone ensemble (normal modes,
short MD, or an AlphaFold/ESMFold ensemble), which is untested here.

Provenance: features from the sigma sweep (commit `d375dac`; Tsuboyama 2023 Dataset1,
Zenodo 7844779; per-domain AlphaFold backbones; `bidirectional_mlm`; ribo Quadro RTX 5000);
re-fit offline with `bayes_design.experiments.calibrate_dg` helpers at commit `4161c16`.

### Visualizing distribution shift
```
bayes-design-experiment viz_probs --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34 --sequence KLPPGWEKRMSRSSGRVYYFNHITNASQFERPSG --results_path ./results/probs_viz_KLPP
```
```
bayes-design-experiment viz_probs --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34 --sequence KLPPGWEKRMSRSSGRVYYFNHITNASQFERPSG --results_path ./results/probs_viz_KLPP_bayes_002 --bayes_balance_factor .002
```
### Evaluate perplexity
bayes-design-experiment compare_seq_metric --protein_id 1PIN --fixed_positions 34 34 --model_name protein_mpnn --metric perplexity --sequences KLPPGWEKRMSRSSGRVYYFNHITNASQFERPSG ELPEGWEERVDPKTGKKIYYNTKTKEITEEKPVK KLPEGWVEKVDPKTGKKVYYNTKTKEITEEKPIG
```
bayes-design-experiment compare_seq_metric --protein_id 1PIN --fixed_positions 34 34 --model_name protein_mpnn --metric perplexity --sequences KLPPGWEKRMSRSSGRVYYFNHITNASQFERPSG
```

# Compare redesign design, from scratch design, wild type
```
bayes-design-experiment compare_seq_metric --protein_id 1PIN --decode_order n_to_c --model_name bayes_design --metric log_prob --sequences QLPEGWEEKVDEETKEKIYYNKETKEITKEKMIC MLPEGWVKQRNPITGEDVCFNTLTHEMTKFEPQG KLPPGWEKRMSRSSGRVYYFNHITNASQFERPSG
```

One problem with using mlm redesign: Even if the probability increases for one token, that probability could be changed once we replace other tokens. So maybe V instead of K increases probability by .01, and then G instead of L increases probability by .01, but decreases probability of V by .2. Perhaps we should instead choose each token based on how it affects the log-likelihood of the whole sequence instead of just that token. argmax_aa p(struct=X|seq=seq:seq[i]=aa) vs argmax_aa p(struct=X|seq)

This:   p(seq:seq[i]=aa|struct)/p(seq:seq[i]=aa)
        = p(seq[i]=aa|struct, seq[!=i]=seq) p(seq[!=i]=seq|struct) /
           ( p(seq[i]=aa|seq[!=i]=seq) p(seq[!=i]=seq) )

Not this: p(seq[i]=aa|struct, seq[!=i]=seq)/p(seq[i]=aa|seq[!=i]=seq)

# Price sequences - 1PIN (beta sheet) aka WW

From scratch (bidirectional autoregressive) log probs:
original sequence               KLPPGWEKRMSRSSGRVYYFNHITNASQFERPSG  -111.97058678054292

BayesDesign
greedy design from scratch,     TLPEHWVKRKDPKTGQWIYENTKTHETLAQKWQG  -40.06295947692667
beam 128 design from scratch,   QLPEGWVKRTNKVTGKDEYRNVKTNETTSKKPIG  -31.762047139937724
beam 256 design from scratch,   QLPEGWVKRTNKVTGKDEYRNVKTNETTSKKPIG  -31.762047139937724
beam 512 design from scratch,   QLPEGWVKRTNKVTGKDEYRNVKTNETTSKKPIG  -31.762047139937724

ProteinMPNN
greedy design from scratch,     TLPEGWVEVVDPETGEKKYYNTKTKEVTSEKPVG  
beam 128 design from scratch,   KLPEGWVEKVDPKTGKKVYYNTKTKEITEEKPVG  
beam 256 design from scratch,   KLPEGWVEKVDPKTGKKVYYNTKTKEITEEKPVG  
beam 512 design from scratch,   KLPEGWVEKVDPKTGEKVYYNTKTKEITKEKPVG  


# Price sequences 1COI (coiled coil) aka 1CW
<!-- From scratch (bidirectional autoregressive) log probs:
original sequence               KLPPGWEKRMSRSSGRVYYFNHITNASQFERPSG

BayesDesign
greedy design from scratch,     TLPEHWVKRKDPKTGQWIYENTKTHETLAQKWQG
beam 128 design from scratch,   QLPEGWVKRTNKVTGKDEYRNVKTNETTSKKPIG
beam 256 design from scratch,   QLPEGWVKRTNKVTGKDEYRNVKTNETTSKKPIG
beam 512 design from scratch,   QLPEGWVKRTNKVTGKDEYRNVKTNETTSKKPIG

ProteinMPNN
greedy design from scratch,     TLPEGWVEVVDPETGEKKYYNTKTKEVTSEKPVG
beam 128 design from scratch,   KLPEGWVEKVDPKTGKKVYYNTKTKEITEEKPVG  
beam 256 design from scratch,   KLPEGWVEKVDPKTGKKVYYNTKTKEITEEKPVG  
beam 512 design from scratch,   KLPEGWVEKVDPKTGEKVYYNTKTKEITKEKPVG   -->

## Price sequence design commands
Greedy design from scratch BayesDesign
```
bayes-design --model_name bayes_design --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34 --bayes_balance_factor .002 --decode_algorithm greedy --from_scratch
```
Beam 128 design from scratch BayesDesign
```
bayes-design --model_name bayes_design --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34 --bayes_balance_factor .002  --decode_algorithm beam_medium --n_beams 128 --from_scratch
```
Beam 256 design from scratch BayesDesign
```
bayes-design --model_name bayes_design --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34 --bayes_balance_factor .002  --decode_algorithm beam_medium --n_beams 256 --from_scratch
```
Beam 512 design from scratch BayesDesign
```
bayes-design --model_name bayes_design --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34 --bayes_balance_factor .002  --decode_algorithm beam_medium --n_beams 512 --from_scratch
```
Greedy redesign BayesDesign
```
bayes-design --model_name bayes_design --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34 --bayes_balance_factor .002 --decode_algorithm greedy
```
Beam 128 redesign BayesDesign
```
bayes-design --model_name bayes_design --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34 --bayes_balance_factor .002 --decode_algorithm beam_medium --n_beams 128
```
Beam 256 redesign BayesDesign
```
bayes-design --model_name bayes_design --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34 --bayes_balance_factor .002 --decode_algorithm beam_medium --n_beams 256
```
Beam 512 redesign BayesDesign
```
bayes-design --model_name bayes_design --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34 --bayes_balance_factor .002 --decode_algorithm beam_medium --n_beams 512
```
Greedy design from scratch ProteinMPNN
```
bayes-design --model_name protein_mpnn --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34  --decode_algorithm greedy --from_scratch
```
Beam 128 design from scratch ProteinMPNN
```
bayes-design --model_name protein_mpnn --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34   --decode_algorithm beam_medium --n_beams 128 --from_scratch
```
Beam 256 design from scratch ProteinMPNN
```
bayes-design --model_name protein_mpnn --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34   --decode_algorithm beam_medium --n_beams 256 --from_scratch
```
Beam 512 design from scratch ProteinMPNN
```
bayes-design --model_name protein_mpnn --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34   --decode_algorithm beam_medium --n_beams 512 --from_scratch
```
Greedy redesign ProteinMPNN
```
bayes-design --model_name protein_mpnn --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34  --decode_algorithm greedy
```
Beam 128 redesign ProteinMPNN
```
bayes-design --model_name protein_mpnn --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34  --decode_algorithm beam_medium --n_beams 128
```
Beam 256 redesign ProteinMPNN
```
bayes-design --model_name protein_mpnn --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34 --decode_algorithm beam_medium --n_beams 256
```
Beam 512 redesign ProteinMPNN
```
bayes-design --model_name protein_mpnn --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34 --decode_algorithm beam_medium --n_beams 512
```

1CW

Greedy design from scratch BayesDesign
```
bayes-design --model_name bayes_design --protein_id 1coi --decode_order n_to_c --fixed_positions 29 29 --bayes_balance_factor .002 --decode_algorithm greedy --from_scratch
```
Beam 128 design from scratch BayesDesign
```
bayes-design --model_name bayes_design --protein_id 1coi --decode_order n_to_c --fixed_positions 29 29 --bayes_balance_factor .002  --decode_algorithm beam_medium --n_beams 128 --from_scratch
```
Beam 256 design from scratch BayesDesign
```
bayes-design --model_name bayes_design --protein_id 1coi --decode_order n_to_c --fixed_positions 29 29 --bayes_balance_factor .002  --decode_algorithm beam_medium --n_beams 256 --from_scratch
```
Beam 512 design from scratch BayesDesign
```
bayes-design --model_name bayes_design --protein_id 1coi --decode_order n_to_c --fixed_positions 29 29 --bayes_balance_factor .002  --decode_algorithm beam_medium --n_beams 512 --from_scratch
```

Greedy design from scratch ProteinMPNN
```
bayes-design --model_name protein_mpnn --protein_id 1coi --decode_order n_to_c --fixed_positions 29 29  --decode_algorithm greedy --from_scratch
```
Beam 128 design from scratch ProteinMPNN
```
bayes-design --model_name protein_mpnn --protein_id 1coi --decode_order n_to_c --fixed_positions 29 29   --decode_algorithm beam_medium --n_beams 128 --from_scratch
```
Beam 256 design from scratch ProteinMPNN
```
bayes-design --model_name protein_mpnn --protein_id 1coi --decode_order n_to_c --fixed_positions 29 29   --decode_algorithm beam_medium --n_beams 256 --from_scratch
```
Beam 512 design from scratch ProteinMPNN
```
bayes-design --model_name protein_mpnn --protein_id 1coi --decode_order n_to_c --fixed_positions 29 29   --decode_algorithm beam_medium --n_beams 512 --from_scratch
```

## Price sequence evaluation commands
bayes-design-experiment compare_seq_metric --model_name bayes_design --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34 --bayes_balance_factor .002 --metric log_prob --sequences KLPPGWEKRMSRSSGRVYYFNHITNASQFERPSG TLPEHWVKRKDPKTGQWIYENTKTHETLAQKWQG KTPEWWWPIINKWTMETMYYNTGTNEVTKEKPIG QLPEGWVKRTNKVTGKDEYRNVKTNETTSKKPIG KTWYGWVPIVDFKTGEEMYRNDLTNEITRDKPIG QTWYGWVPIVDDKTGETKWLNKIEKKVTSKKPIG QWWYGWVPIVDEKTGEEKAYNVLTKEVTSERPIG

bayes-design-experiment compare_seq_metric --model_name bayes_design --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34 --bayes_balance_factor .002 --metric log_prob --sequences KLPPGWEKRMSRSSGRVYYFNHITNASQFERPSG TLPEHWVKRKDPKTGQWIYENTKTHETLAQKWQG KTPEWWWPIINKWTMETMYYNTGTNEVTKEKPIG QLPEGWVKRTNKVTGKDEYRNVKTNETTSKKPIG KTWYGWVPIVDFKTGEEMYRNDLTNEITRDKPIG QTWYGWVPIVDDKTGETKWLNKIEKKVTSKKPIG QWWYGWVPIVDEKTGEEKAYNVLTKEVTSERPIG --from_scratch

# Bundy sequences
Fix all residues except 16 engineered residues, greedy decode redesign
```
bayes-design --model_name bayes_design --protein_id nanoluc --decode_order proximity --bayes_balance_factor .002 --decode_algorithm greedy --fixed_positions 1 9 10 13 15 20 22 27 29 36 38 42 44 53 55 63 65 77 79 81 83 84 86 99 101 124 126  133 135 147 149 175 177 179
```
Fix the active site (small), greedy decode redesign
```
bayes-design --model_name bayes_design --protein_id nanoluc --decode_order proximity --bayes_balance_factor .002 --decode_algorithm greedy --fixed_positions 1 9 10 10 13 13 51 51 59 59 61 61 63 63 100 102 104 104 106 106 112 112 114 114 121 125 127 127 132 132 134 137 144 144 146 146 148 148 156 156 158 158 160 160
```
Fix the active site (small), beam search decode from scratch
```
bayes-design --model_name bayes_design --protein_id nanoluc --decode_order proximity --bayes_balance_factor .002 --decode_algorithm beam_medium --n_beams 512 --from_scratch --fixed_positions  1 9 10 10 13 13 51 51 59 59 61 61 63 63 100 102 104 104
```
Fix the active site (small) and the engineered residues, greedy decode redesign
```
bayes-design --model_name bayes_design --protein_id nanoluc --decode_order proximity --bayes_balance_factor .002 --decode_algorithm greedy --fixed_positions 1 9 10 10 13 13 51 51 59 59 61 61 63 63 100 102 104 104 106 106 112 112 114 114 121 125 127 127 132 132 134 137 144 144 146 146 148 148 156 156 158 158 160 160 14 14 21 21 28 28 37 37 43 43 53 54 64 64 78 78 82 82 85 85 100 100 125 125 134 134 148 148 176 176
```
Fix the active site (small) and the engineered residues, beam search decode from scratch
```
bayes-design --model_name bayes_design --protein_id nanoluc --decode_order proximity --bayes_balance_factor .002 --decode_algorithm beam_medium --n_beams 512 --from_scratch --fixed_positions  1 9 10 10 13 13 51 51 59 59 61 61 63 63 100 102 104 104 106 106 112 112 114 114 121 125 127 127 132 132 134 137 144 144 146 146 148 148 156 156 158 158 160 160 14 14 21 21 28 28 37 37 43 43 53 54 64 64 78 78 82 82 85 85 100 100 125 125 134 134 148 148 176 176
```
Fix the active site (large), greedy decode redesign
```
bayes-design --model_name bayes_design --protein_id nanoluc --decode_order proximity --bayes_balance_factor .002 --decode_algorithm greedy --fixed_positions 1 9 8 14 17 17 49 51 57 64 98 108 110 116 120 127 132 139 142 150 156 158 160 162 
```
Fix the active site (large), beam search decode from scratch
```
bayes-design --model_name bayes_design --protein_id nanoluc --decode_order proximity --bayes_balance_factor .002 --decode_algorithm beam_medium --n_beams 512 --from_scratch --fixed_positions 1 9 8 14 17 17 49 51 57 64 98 108 110 116 120 127 132 139 142 150 156 158 160 162
```
Fix the active site (large) and the engineered residues, greedy decode redesign
```
bayes-design --model_name bayes_design --protein_id nanoluc --decode_order proximity --bayes_balance_factor .002 --decode_algorithm greedy --fixed_positions 1 9 8 14 17 17 49 51 57 64 98 108 110 116 120 127 132 139 142 150 156 158 160 162 14 14 21 21 28 28 37 37 43 43 53 54 64 64 78 78 82 82 85 85 100 100 125 125 134 134 148 148 176 176 
```
Fix the active site (large), and the engineered residues, beam search decode from scratch
```
bayes-design --model_name bayes_design --protein_id nanoluc --decode_order proximity --bayes_balance_factor .002 --decode_algorithm beam_medium --n_beams 512 --from_scratch --fixed_positions 1 9 8 14 17 17 49 51 57 64 98 108 110 116 120 127 132 139 142 150 156 158 160 162 14 14 21 21 28 28 37 37 43 43 53 54 64 64 78 78 82 82 85 85 100 100 125 125 134 134 148 148 176 176 
```

bayes-design-experiment compare_seq_metric --model_name bayes_design --protein_id nanoluc --decode_order proximity --bayes_balance_factor .002 --metric log_prob --sequences MWSHPQFEKVFTLEDFVGDWRQTAGYNLDQVLEQGGVSSLFQNLGVSVTPIQRIVLSGENGLKIDIHVIIPYEGLSGDQMGQIEKIFKVVYPVDDHHFKVILHYGTLVIDGVTPNMIDYFGRPYEGIAVFDGKKITVTGTLWNGNKIIDERLINPDGSLLFRVTINGVTGWRLCERILA MWSHPQFEKVFTLEDFVGDWRQTAGYNMDQVLEQGGASSLFQKLGVSVTPIQRIVLSGENGLKVDIHVIIPYEGLSGCQMGLIEKIFKVVYPVDDHHFKVILHYGTLVIDGVTPNMIDYFGRPYTGIAVFDGKKITVTGTLWNGNKIVDERLINPDGSLLFRVTINGVTGWRLCERILA --fixed_positions 1 9 10 13 15 20 22 27 29 36 38 42 44 53 55 63 65 77 79 81 83 84 86 99 101 124 126 133 135 147 149 175 177 179

bayes-design-experiment compare_seq_metric --model_name bayes_design --protein_id nanoluc --decode_order proximity --bayes_balance_factor .002 --metric log_prob --sequences MWSHPQFEKVFTLEDFVGDWRQTAGYNLDQVLEQGGVSSLFQNLGVSVTPIQRIVLSGENGLKIDIHVIIPYEGLSGDQMGQIEKIFKVVYPVDDHHFKVILHYGTLVIDGVTPNMIDYFGRPYEGIAVFDGKKITVTGTLWNGNKIIDERLINPDGSLLFRVTINGVTGWRLCERILA MWSHPQFEKVLTLDDFVGKWELVEQKNIPEVLRQMGAPQFFIDLWCNTKPILVITKYGENGLKVTIEMVIPKKGLTCDQMDQIHKIFKVMIPVDENHFKVILDYGTLIINGVSPNCKDWLGRPYEGICTFDGKKITVTGTLPNGNKFIDTYEILPDGSLLFTVDVNGVKGWWKLKRVEE --fixed_positions 1 9 10 10 13 13 51 51 59 59 61 61 63 63 100 102 104 104 106 106 112 112 114 114 121 125 127 127 132 132 134 137 144 144 146 146 148 148 156 156 158 158 160 160

bayes-design-experiment compare_seq_metric --model_name bayes_design --protein_id nanoluc --decode_order proximity --bayes_balance_factor .002 --metric log_prob --from_scratch --sequences MWSHPQFEKVFTLEDFVGDWRQTAGYNLDQVLEQGGVSSLFQNLGVSVTPIQRIVLSGENGLKIDIHVIIPYEGLSGDQMGQIEKIFKVVYPVDDHHFKVILHYGTLVIDGVTPNMIDYFGRPYEGIAVFDGKKITVTGTLWNGNKIIDERLINPDGSLLFRVTINGVTGWRLCERILA MWSHPQFEKVLTLDDFVGNWRMVSQWNIPAVLREMGMPPFLIDLWCATTPIWVITKYGENGLKVDVHMVIPKEGLTPEQMRYLQAMFGHMTQVDETHFQVILDYGVFIINGTSKNCKDFMNRPFEVNTTFDGKKLTMTGTLWNGKKFVMTFEILPDGHLRYTVDVNGVKGWMILERVEP --fixed_positions 1 9 10 10 13 13 51 51 59 59 61 61 63 63 100 102 104 104 106 106 112 112 114 114 121 125 127 127 132 132 134 137 144 144 146 146 148 148 156 156 158 158 160 160

bayes-design-experiment compare_seq_metric --model_name bayes_design --protein_id nanoluc --decode_order proximity --bayes_balance_factor .002 --metric log_prob --sequences MWSHPQFEKVFTLEDFVGDWRQTAGYNLDQVLEQGGVSSLFQNLGVSVTPIQRIVLSGENGLKIDIHVIIPYEGLSGDQMGQIEKIFKVVYPVDDHHFKVILHYGTLVIDGVTPNMIDYFGRPYEGIAVFDGKKITVTGTLWNGNKIIDERLINPDGSLLFRVTINGVTGWRLCERILA MWSHPQFEKVLTLEDFVGDWRLVDKWNLPEVLKAMGVPQFMINLYCQTQPILRITKAGENGLKIEIEMVIPKKGLTCDQMEQIKKIYKHVEDVDDNHFKVILDYGTLIINGVSPNMKDFLGRPYEGICTFDGKKITVTGTLPNGNKVIITFEIQPDGSLLLTIDVNGVKGWMVYERVEP --fixed_positions 1 9 10 10 13 13 51 51 59 59 61 61 63 63 100 102 104 104 106 106 112 112 114 114 121 125 127 127 132 132 134 137 144 144 146 146 148 148 156 156 158 158 160 160 14 14 21 21 28 28 37 37 43 43 53 54 64 64 78 78 82 82 85 85 100 100 125 125 134 134 148 148 176 176

bayes-design-experiment compare_seq_metric --model_name bayes_design --protein_id nanoluc --decode_order proximity --bayes_balance_factor .002 --metric log_prob --from_scratch --sequences MWSHPQFEKVFTLEDFVGDWRQTAGYNLDQVLEQGGVSSLFQNLGVSVTPIQRIVLSGENGLKIDIHVIIPYEGLSGDQMGQIEKIFKVVYPVDDHHFKVILHYGTLVIDGVTPNMIDYFGRPYEGIAVFDGKKITVTGTLWNGNKIIDERLINPDGSLLFRVTINGVTGWRLCERILA MWSHPQFEKVLKLEDFVGDWRRVDSWNLPEVLKAMGVPQFFINLFCQTQPIWRISKHGEKGLKIQMIMRIPKQGLTPDQMAQIQKTFKHVQDIDDQHFQVILDYGTLIIDGVSPNCKDFLGRPYEGICKFDGKKITVTGTLPNGNKFIWTMEILDDGSLLFTVDVNGVKGYMILERVEP --fixed_positions 1 9 10 10 13 13 51 51 59 59 61 61 63 63 100 102 104 104 106 106 112 112 114 114 121 125 127 127 132 132 134 137 144 144 146 146 148 148 156 156 158 158 160 160 14 14 21 21 28 28 37 37 43 43 53 54 64 64 78 78 82 82 85 85 100 100 125 125 134 134 148 148 176 176

bayes-design-experiment compare_seq_metric --model_name bayes_design --protein_id nanoluc --decode_order proximity --bayes_balance_factor .002 --metric log_prob --sequences MWSHPQFEKVFTLEDFVGDWRQTAGYNLDQVLEQGGVSSLFQNLGVSVTPIQRIVLSGENGLKIDIHVIIPYEGLSGDQMGQIEKIFKVVYPVDDHHFKVILHYGTLVIDGVTPNMIDYFGRPYEGIAVFDGKKITVTGTLWNGNKIIDERLINPDGSLLFRVTINGVTGWRLCERILA MWSHPQFEKVFTLEDFVGKWRMVSKQNTPAVLKEEGAPQFLIDLWCNTTPIFIITLSGENGLKIDIEMIIPKKGLTCDQMKYLQKIFKVMIPVDENNFKVILHYGTLVIDGVTPNMKDYFGRPYEGICKFDGKKITVTGTLWNGNKIIDEWEILPDGSLLFRRTVNGVVGWWKLERIEE --fixed_positions 1 9 8 14 17 17 49 51 57 64 98 108 110 116 120 127 132 139 142 150 156 158 160 162

bayes-design-experiment compare_seq_metric --model_name bayes_design --protein_id nanoluc --decode_order proximity --bayes_balance_factor .002 --metric log_prob --from_scratch --sequences MWSHPQFEKVFTLEDFVGDWRQTAGYNLDQVLEQGGVSSLFQNLGVSVTPIQRIVLSGENGLKIDIHVIIPYEGLSGDQMGQIEKIFKVVYPVDDHHFKVILHYGTLVIDGVTPNMIDYFGRPYEGIAVFDGKKITVTGTLWNGNKIIDERLINPDGSLLFRVTINGVTGWRLCERILA MWSHPQFEKVFTLEDFVGDWRLVSKQNMAAVLREMGAPDFLIQLYLQCTPIFHITKSGENGLKIDVEMIIPKAGLTPEQMCYLQKMFKHMEPVDENHFKVILHYGTLVIDGVTPNMKDAFGRPYEGICKFDGKKITVTGTLWNGNKIIDEYEILPDGSLLFRRTVNGVTGWMKLERVEP --fixed_positions 1 9 8 14 17 17 49 51 57 64 98 108 110 116 120 127 132 139 142 150 156 158 160 162

bayes-design-experiment compare_seq_metric --model_name bayes_design --protein_id nanoluc --decode_order proximity --bayes_balance_factor .002 --metric log_prob --sequences MWSHPQFEKVFTLEDFVGDWRQTAGYNLDQVLEQGGVSSLFQNLGVSVTPIQRIVLSGENGLKIDIHVIIPYEGLSGDQMGQIEKIFKVVYPVDDHHFKVILHYGTLVIDGVTPNMIDYFGRPYEGIAVFDGKKITVTGTLWNGNKIIDERLINPDGSLLFRVTINGVTGWRLCERILA MWSHPQFEKVFTLEDFVGDWRMVKQWNLPAVLKAMGVPQFMINLFCQTTPILRITLSGENGLKIDIEMIIPKKGLTCDQMNQIKKIFKHVEDVDDNNFKVILHYGTLVIDGVTPNMKDWFGRPYEGICKFDGKKITVTGTLWNGNKIIDEFEILPDGSLLFRVTVNGVEGWMIYERVEP --fixed_positions 1 9 8 14 17 17 49 51 57 64 98 108 110 116 120 127 132 139 142 150 156 158 160 162 14 14 21 21 28 28 37 37 43 43 53 54 64 64 78 78 82 82 85 85 100 100 125 125 134 134 148 148 176 176

bayes-design-experiment compare_seq_metric --model_name bayes_design --protein_id nanoluc --decode_order proximity --bayes_balance_factor .002 --metric log_prob --from_scratch --sequences MWSHPQFEKVFTLEDFVGDWRQTAGYNLDQVLEQGGVSSLFQNLGVSVTPIQRIVLSGENGLKIDIHVIIPYEGLSGDQMGQIEKIFKVVYPVDDHHFKVILHYGTLVIDGVTPNMIDYFGRPYEGIAVFDGKKITVTGTLWNGNKIIDERLINPDGSLLFRVTINGVTGWRLCERILA MWSHPQFEKVFTLEDFVGDWREVDRWNLADVLKAMGVPQFLINLYMSCTPIWRITKSGENGLKIDVEMIIPKQGLTEDQLQQIKKIFQHVEDVDDNHFKVILHYGTLVIDGVTPNMKDWFGRPYEGICKFDGKKITVTGTLWNGNKIIDEFEILPDGSLLFRTTVNGVTGYRILERVEP --fixed_positions 1 9 8 14 17 17 49 51 57 64 98 108 110 116 120 127 132 139 142 150 156 158 160 162 14 14 21 21 28 28 37 37 43 43 53 54 64 64 78 78 82 82 85 85 100 100 125 125 134 134 148 148 176 176

# Moody lab design

# Moody sequences
From scratch (bidirectional autoregressive) log probs:
original sequence               SATHIKFSKRDEDGKELAGATMELRDSSGKTISTWISDGQVKDFYLYPGKYTFVETAAPDGYEVATAITFGHHHHHHHHHH

BayesDesign
greedy design from scratch,     MPHEVRYEKRDSNGRLVKGYNWSLLDAEFNVIATWVSDGKPKYFMLPDGIYTWVETQAPPGYPKQPDVTFGHHHHHHHHHH
beam 128 design from scratch,   GKQHVEIRKVDKNGRLLKGAKWELLNSEGDVIDQWISDGRPKHFWLPWGHYTLRETEPMPGFPKRPPETFGHHHHHHHHHH
beam 256 design from scratch,   
beam 512 design from scratch,   

ProteinMPNN                     
greedy design from scratch,     SLEKVTIKKVDSNGNLLSGAKWELLDENGNVIKTWTSDGKPKEFELPPGIYTVKETEAPAGYSKAADETFGHHHHHHHHHH
beam 128 design from scratch,   EKKKVTIEKKDENGNLLKGAKWELLNEKGEVIEEWTSDGKPKEFELPEGIYTVKETEAPEGYEKKEPETFGHHHHHHHHHH
beam 256 design from scratch,   
beam 512 design from scratch,   
  
Greedy design from scratch BayesDesign
```[]
bayes-design --model_name bayes_design --protein_id spycatcher --decode_order n_to_c --fixed_positions 9 9 55 55 72 81 --bayes_balance_factor .002 --decode_algorithm greedy --from_scratch
```
Beam 128 design from scratch BayesDesign
```
bayes-design --model_name bayes_design --protein_id spycatcher --decode_order n_to_c --fixed_positions 9 9 55 55 72 81 --bayes_balance_factor .002  --decode_algorithm beam_medium --n_beams 128 --from_scratch
```
Beam 256 design from scratch BayesDesign
```
bayes-design --model_name bayes_design --protein_id spycatcher --decode_order n_to_c --fixed_positions 9 9 55 55 72 81 --bayes_balance_factor .002  --decode_algorithm beam_medium --n_beams 256 --from_scratch
```
Beam 512 design from scratch BayesDesign
```
bayes-design --model_name bayes_design --protein_id spycatcher --decode_order n_to_c --fixed_positions 9 9 55 55 72 81 --bayes_balance_factor .002  --decode_algorithm beam_medium --n_beams 512 --from_scratch
```

Greedy design from scratch ProteinMPNN
```
bayes-design --model_name protein_mpnn --protein_id spycatcher --decode_order n_to_c --fixed_positions 9 9 55 55 72 81 --decode_algorithm greedy --from_scratch
```
Beam 128 design from scratch ProteinMPNN
```
bayes-design --model_name protein_mpnn --protein_id spycatcher --decode_order n_to_c --fixed_positions 9 9 55 55 72 81 --decode_algorithm beam_medium --n_beams 128 --from_scratch
```
Beam 256 design from scratch ProteinMPNN
```
bayes-design --model_name protein_mpnn --protein_id spycatcher --decode_order n_to_c --fixed_positions 9 9 55 55 72 81 --decode_algorithm beam_medium --n_beams 256 --from_scratch
```
Beam 512 design from scratch ProteinMPNN
```
bayes-design --model_name protein_mpnn --protein_id spycatcher --decode_order n_to_c --fixed_positions 9 9 55 55 72 81 --decode_algorithm beam_medium --n_beams 512 --from_scratch
```

# SpyCatcher Extended

DSATHIKFSKRDIDGKELAGATMELRDSSGKTISTWISDGQVKDFYLMPGKYTFVETAAPDGYEVATAITFTVNEQGQVTVNGHHHHHHHHHH

BayesDesign
greedy design from scratch,     

ProteinMPNN                     
greedy design from scratch,     

Greedy design from scratch BayesDesign
```
bayes-design --model_name bayes_design --protein_id spycatcher_long --decode_order n_to_c --fixed_positions 10 10 56 56 83 93 --bayes_balance_factor .002 --decode_algorithm greedy
```
Greedy design from scratch ProteinMPNN
```
bayes-design --model_name protein_mpnn --protein_id spycatcher_long --decode_order n_to_c --fixed_positions 10 10 56 56 83 93 --bayes_balance_factor .002 --decode_algorithm greedy
```

# Steered evolution
bayes-design --model_name bayes_design --protein_id nanoluc --decode_order n_to_c --decode_algorithm combinations --fixed_positions 1 47 54 83 93 167 172 179