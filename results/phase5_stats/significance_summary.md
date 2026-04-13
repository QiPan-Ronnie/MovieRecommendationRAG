# Phase 5 Significance Summary

## Dataset Sources

- `hybrid`: `/root/autodl-tmp/MovieRecommendationRAG/results/phase5_with_recommendation_Hybrid_bertscore_unified/faithfulness_rag/faithfulness_detailed.jsonl`
- `hybrid_perturb`: `/root/autodl-tmp/MovieRecommendationRAG/results/phase5_with_recommendation_Hybrid_p500/faithfulness_perturbation/faithfulness_detailed.jsonl`
- `hybrid_prompt`: `/root/autodl-tmp/MovieRecommendationRAG/results/phase5_with_recommendation_Hybrid_bertscore_unified/faithfulness_prompt_only/faithfulness_detailed.jsonl`
- `kg_only`: `/root/autodl-tmp/MovieRecommendationRAG/results/phase5_with_recommendation_KG_Only_bertscore_unified/faithfulness_kg_only/faithfulness_detailed.jsonl`
- `kg_perturb`: `/root/autodl-tmp/MovieRecommendationRAG/results/phase5_with_recommendation_KG_Only_p500/faithfulness_perturbation/faithfulness_detailed.jsonl`
- `kg_prompt`: `/root/autodl-tmp/MovieRecommendationRAG/results/phase5_with_recommendation_KG_Only_bertscore_unified/faithfulness_prompt_only/faithfulness_detailed.jsonl`
- `retrieval_only`: `/root/autodl-tmp/MovieRecommendationRAG/results/phase5_with_recommendation_Retrieval_Only_bertscore_unified/faithfulness_rag/faithfulness_detailed.jsonl`
- `retrieval_perturb`: `/root/autodl-tmp/MovieRecommendationRAG/results/phase5_with_recommendation_Retrieval_Only_p500/faithfulness_perturbation/faithfulness_detailed.jsonl`
- `retrieval_prompt`: `/root/autodl-tmp/MovieRecommendationRAG/results/phase5_with_recommendation_Retrieval_Only_bertscore_unified/faithfulness_prompt_only/faithfulness_detailed.jsonl`

## Main Comparisons

| left_label     | right_label      | metric           | pair_count | left_mean | right_mean | mean_difference | ci_low     | ci_high    | t_statistic | p_value     |
| -------------- | ---------------- | ---------------- | ---------- | --------- | ---------- | --------------- | ---------- | ---------- | ----------- | ----------- |
| hybrid         | retrieval_only   | evidence_overlap | 59500      | 0.202383  | 0.145212   | 0.0571714       | 0.0565073  | 0.0578403  | 177.065     | 0           |
| hybrid         | retrieval_only   | rouge_l          | 59500      | 0.234486  | 0.167443   | 0.0670425       | 0.0665164  | 0.0675676  | 252.927     | 0           |
| hybrid         | retrieval_only   | semantic_sim     | 59500      | 0.908844  | 0.887083   | 0.0217615       | 0.0215383  | 0.0219835  | 187.286     | 0           |
| hybrid         | retrieval_only   | bert_score_f1    | 59500      | 0.844269  | 0.839103   | 0.00516563      | 0.00503792 | 0.00529454 | 80.7514     | 0           |
| hybrid         | kg_only          | evidence_overlap | 59481      | 0.202407  | 0.200712   | 0.00169478      | 0.00108739 | 0.00226231 | 5.46611     | 4.61857e-08 |
| hybrid         | kg_only          | rouge_l          | 59481      | 0.234504  | 0.211201   | 0.0233024       | 0.0227571  | 0.0238167  | 83.6596     | 0           |
| hybrid         | kg_only          | semantic_sim     | 59481      | 0.908846  | 0.94515    | -0.0363038      | -0.036526  | -0.0360803 | -311.204    | 0           |
| hybrid         | kg_only          | bert_score_f1    | 59481      | 0.844273  | 0.883459   | -0.0391866      | -0.0393229 | -0.0390572 | -568.129    | 0           |
| hybrid         | hybrid_prompt    | evidence_overlap | 59500      | 0.202383  | 0.0967833  | 0.1056          | 0.105032   | 0.106191   | 346.19      | 0           |
| hybrid         | hybrid_prompt    | rouge_l          | 59500      | 0.234486  | 0.170535   | 0.0639505       | 0.0634958  | 0.0644155  | 258.311     | 0           |
| hybrid         | hybrid_prompt    | semantic_sim     | 59500      | 0.908844  | 0.946159   | -0.037314       | -0.0374917 | -0.037139  | -408.452    | 0           |
| hybrid         | hybrid_prompt    | bert_score_f1    | 59500      | 0.844269  | 0.813477   | 0.0307921       | 0.0306891  | 0.0308931  | 570.877     | 0           |
| retrieval_only | retrieval_prompt | evidence_overlap | 59500      | 0.145212  | 0.0861466  | 0.059065        | 0.0585658  | 0.0595882  | 225.115     | 0           |
| retrieval_only | retrieval_prompt | rouge_l          | 59500      | 0.167443  | 0.136782   | 0.0306609       | 0.0302809  | 0.0310249  | 156.036     | 0           |
| retrieval_only | retrieval_prompt | semantic_sim     | 59500      | 0.887083  | 0.924397   | -0.0373141      | -0.0374849 | -0.0371496 | -446.75     | 0           |
| retrieval_only | retrieval_prompt | bert_score_f1    | 59500      | 0.839103  | 0.819618   | 0.0194853       | 0.0193888  | 0.0195766  | 410.135     | 0           |
| kg_only        | kg_prompt        | evidence_overlap | 59481      | 0.200712  | 0.0440778  | 0.156634        | 0.155961   | 0.157296   | 474.723     | 0           |
| kg_only        | kg_prompt        | rouge_l          | 59481      | 0.211201  | 0.106262   | 0.104939        | 0.104368   | 0.10549    | 371.155     | 0           |
| kg_only        | kg_prompt        | semantic_sim     | 59481      | 0.94515   | 0.969545   | -0.0243946      | -0.0245375 | -0.0242563 | -334.154    | 0           |
| kg_only        | kg_prompt        | bert_score_f1    | 59481      | 0.883459  | 0.827897   | 0.0555619       | 0.0554126  | 0.05571    | 718.405     | 0           |
| kg_only        | retrieval_only   | evidence_overlap | 59481      | 0.200712  | 0.145209   | 0.0555025       | 0.0547346  | 0.0562409  | 145.242     | 0           |
| kg_only        | retrieval_only   | rouge_l          | 59481      | 0.211201  | 0.167444   | 0.0437569       | 0.0430891  | 0.0444237  | 131.168     | 0           |
| kg_only        | retrieval_only   | semantic_sim     | 59481      | 0.94515   | 0.887083   | 0.058067        | 0.0578353  | 0.0582877  | 493.549     | 0           |
| kg_only        | retrieval_only   | bert_score_f1    | 59481      | 0.883459  | 0.839103   | 0.0443567       | 0.0441772  | 0.0445222  | 506.265     | 0           |

## Perturbation Comparisons

| left_label        | right_label       | metric           | pair_count | left_mean | right_mean | mean_difference | ci_low       | ci_high     | t_statistic | p_value      | dataset        |
| ----------------- | ----------------- | ---------------- | ---------- | --------- | ---------- | --------------- | ------------ | ----------- | ----------- | ------------ | -------------- |
| hybrid:E1         | hybrid:E2         | evidence_overlap | 500        | 0.196203  | 0.1226     | 0.0736024       | 0.0669849    | 0.0808461   | 20.8011     | 1.15195e-69  | hybrid         |
| hybrid:E1         | hybrid:E2         | rouge_l          | 500        | 0.23114   | 0.163149   | 0.067991        | 0.0624876    | 0.0735803   | 23.7106     | 8.5202e-84   | hybrid         |
| hybrid:E1         | hybrid:E2         | semantic_sim     | 500        | 0.9073    | 0.906054   | 0.0012458       | -0.00128169  | 0.0035914   | 0.98891     | 0.323186     | hybrid         |
| hybrid:E1         | hybrid:E2         | bert_score_f1    | 500        | 0.843378  | 0.836715   | 0.006663        | 0.00522645   | 0.0081392   | 8.89915     | 1.04138e-17  | hybrid         |
| hybrid:E1         | hybrid:E3         | evidence_overlap | 500        | 0.196203  | 0.18712    | 0.009083        | 0.0033127    | 0.0146572   | 3.0321      | 0.00255493   | hybrid         |
| hybrid:E1         | hybrid:E3         | rouge_l          | 500        | 0.23114   | 0.22217    | 0.0089698       | 0.004353     | 0.0139276   | 3.5207      | 0.00046986   | hybrid         |
| hybrid:E1         | hybrid:E3         | semantic_sim     | 500        | 0.9073    | 0.907211   | 8.84e-05        | -0.002398    | 0.00269399  | 0.0683264   | 0.945553     | hybrid         |
| hybrid:E1         | hybrid:E3         | bert_score_f1    | 500        | 0.843378  | 0.841962   | 0.0014156       | 0.000229725  | 0.00266487  | 2.3552      | 0.0188988    | hybrid         |
| hybrid:E1         | hybrid:E4         | evidence_overlap | 500        | 0.196203  | 0.0360642  | 0.160139        | 0.153954     | 0.166012    | 51.224      | 7.48398e-201 | hybrid         |
| hybrid:E1         | hybrid:E4         | rouge_l          | 500        | 0.23114   | 0.143169   | 0.0879708       | 0.0821808    | 0.0939659   | 30.3112     | 3.14041e-115 | hybrid         |
| hybrid:E1         | hybrid:E4         | semantic_sim     | 500        | 0.9073    | 0.91146    | -0.0041604      | -0.00689542  | -0.00137138 | -2.97096    | 0.00311181   | hybrid         |
| hybrid:E1         | hybrid:E4         | bert_score_f1    | 500        | 0.843378  | 0.795903   | 0.047475        | 0.0460692    | 0.0488196   | 65.1206     | 4.4985e-246  | hybrid         |
| retrieval_only:E1 | retrieval_only:E2 | evidence_overlap | 500        | 0.135384  | 0.0979224  | 0.0374614       | 0.0315482    | 0.0431729   | 12.7374     | 2.24661e-32  | retrieval_only |
| retrieval_only:E1 | retrieval_only:E2 | rouge_l          | 500        | 0.178     | 0.134478   | 0.0435218       | 0.0393452    | 0.0477863   | 19.887      | 3.06553e-65  | retrieval_only |
| retrieval_only:E1 | retrieval_only:E2 | semantic_sim     | 500        | 0.897589  | 0.910686   | -0.0130968      | -0.015717    | -0.0101834  | -9.14218    | 1.54607e-18  | retrieval_only |
| retrieval_only:E1 | retrieval_only:E2 | bert_score_f1    | 500        | 0.835943  | 0.83886    | -0.0029174      | -0.00419861  | -0.00166962 | -4.45839    | 1.02081e-05  | retrieval_only |
| retrieval_only:E1 | retrieval_only:E3 | evidence_overlap | 500        | 0.135384  | 0.136928   | -0.0015444      | -0.00705459  | 0.00429485  | -0.555908   | 0.578523     | retrieval_only |
| retrieval_only:E1 | retrieval_only:E3 | rouge_l          | 500        | 0.178     | 0.177594   | 0.0004058       | -0.00357049  | 0.00452893  | 0.201686    | 0.840245     | retrieval_only |
| retrieval_only:E1 | retrieval_only:E3 | semantic_sim     | 500        | 0.897589  | 0.898035   | -0.000446       | -0.00250751  | 0.00161521  | -0.410845   | 0.681363     | retrieval_only |
| retrieval_only:E1 | retrieval_only:E3 | bert_score_f1    | 500        | 0.835943  | 0.835483   | 0.0004604       | -0.00044033  | 0.00135278  | 1.0143      | 0.31093      | retrieval_only |
| retrieval_only:E1 | retrieval_only:E4 | evidence_overlap | 500        | 0.135384  | 0.0273846  | 0.107999        | 0.101995     | 0.113935    | 35.3906     | 3.72118e-138 | retrieval_only |
| retrieval_only:E1 | retrieval_only:E4 | rouge_l          | 500        | 0.178     | 0.119121   | 0.0588792       | 0.0538792    | 0.0638561   | 23.043      | 1.48261e-80  | retrieval_only |
| retrieval_only:E1 | retrieval_only:E4 | semantic_sim     | 500        | 0.897589  | 0.911473   | -0.0138836      | -0.0166464   | -0.0111311  | -9.7862     | 8.33459e-21  | retrieval_only |
| retrieval_only:E1 | retrieval_only:E4 | bert_score_f1    | 500        | 0.835943  | 0.800753   | 0.0351898       | 0.0338444    | 0.0365387   | 51.496      | 8.10134e-202 | retrieval_only |
| kg_only:E1        | kg_only:E2        | evidence_overlap | 500        | 0.198366  | 0.192849   | 0.0055176       | -0.000795905 | 0.0118142   | 1.74396     | 0.0817815    | kg_only        |
| kg_only:E1        | kg_only:E2        | rouge_l          | 500        | 0.212481  | 0.192882   | 0.019599        | 0.0131554    | 0.0260703   | 6.04421     | 2.94183e-09  | kg_only        |
| kg_only:E1        | kg_only:E2        | semantic_sim     | 500        | 0.945833  | 0.955061   | -0.009228       | -0.0109389   | -0.00737877 | -10.2594    | 1.54799e-22  | kg_only        |
| kg_only:E1        | kg_only:E2        | bert_score_f1    | 500        | 0.883262  | 0.886933   | -0.003671       | -0.00536366  | -0.00202919 | -4.33268    | 1.78194e-05  | kg_only        |
| kg_only:E1        | kg_only:E3        | evidence_overlap | 500        | 0.198366  | 0.192127   | 0.0062396       | 0.000544285  | 0.0119963   | 2.09484     | 0.0366888    | kg_only        |
| kg_only:E1        | kg_only:E3        | rouge_l          | 500        | 0.212481  | 0.197684   | 0.0147974       | 0.00865422   | 0.0211125   | 4.82949     | 1.82356e-06  | kg_only        |
| kg_only:E1        | kg_only:E3        | semantic_sim     | 500        | 0.945833  | 0.946061   | -0.0002282      | -0.00223574  | 0.00178405  | -0.229905   | 0.81826      | kg_only        |
| kg_only:E1        | kg_only:E3        | bert_score_f1    | 500        | 0.883262  | 0.879682   | 0.0035806       | 0.00216098   | 0.00492205  | 5.04703     | 6.29775e-07  | kg_only        |
| kg_only:E1        | kg_only:E4        | evidence_overlap | 500        | 0.198366  | 0.0641958  | 0.134171        | 0.126421     | 0.142221    | 32.3204     | 1.87498e-124 | kg_only        |
| kg_only:E1        | kg_only:E4        | rouge_l          | 500        | 0.212481  | 0.103241   | 0.109241        | 0.102217     | 0.11645     | 29.5918     | 7.00215e-112 | kg_only        |
| kg_only:E1        | kg_only:E4        | semantic_sim     | 500        | 0.945833  | 0.937218   | 0.0086152       | 0.0062472    | 0.0111253   | 6.76037     | 3.8556e-11   | kg_only        |
| kg_only:E1        | kg_only:E4        | bert_score_f1    | 500        | 0.883262  | 0.835676   | 0.047586        | 0.0453696    | 0.0498701   | 40.9304     | 1.33076e-161 | kg_only        |
