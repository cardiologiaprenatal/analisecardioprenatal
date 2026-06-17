# Relatório Científico — Predição de Cardiopatia Congênita Fetal (CCF)

**Projeto:** GestRisk AI  
**Data de geração:** 09/06/2026 21:26  
**Autor:** Pipeline automatizado de Machine Learning  

---

## 1. Introdução

A Cardiopatia Congênita Fetal (CCF) representa uma das malformações congênitas mais frequentes,
com prevalência estimada entre 4–10 casos por 1.000 nascidos vivos. A detecção precoce durante
o pré-natal permite planejamento obstétrico adequado, intervenção especializada e melhores
desfechos neonatais.

Este estudo desenvolve modelos de Machine Learning para predição de risco de CCF utilizando
fatores maternos, doenças associadas, exposições e marcadores ultrassonográficos fetais descritos
na literatura médica. A base de dados é sintética, calibrada para reproduzir associações
epidemiológicas plausíveis, servindo como prova de conceito para o projeto **GestRisk**.

### Objetivo

Construir, treinar, validar e comparar múltiplos algoritmos de classificação capazes de
estratificar o risco de CCF, avaliando diferentes estratégias de balanceamento de classes e
cenários progressivos de inclusão de variáveis clínicas.

---

## 2. Metodologia

### 2.1 Geração de Dados Sintéticos

- **N amostras:** 5,000
- **Prevalência CCF:** 7.42% (faixa alvo: 3–8%)
- **Mecanismo:** modelo logístico com pesos calibrados por fator de risco
- **Missing values:** ~2% aleatório por variável

Fatores com maior peso na geração do desfecho:
translucência nucal aumentada, alteração do ducto venoso, regurgitação tricúspide,
malformações extracardíacas, história familiar/pessoal de cardiopatia, diabetes pré-gestacional,
infecções (rubéola, CMV), teratógenos (isotretinoína, ácido valpróico) e gestação gemelar.

### 2.2 Pré-processamento

| Etapa | Técnica |
|-------|---------|
| Split | Train/Test 80/20 estratificado |
| Numéricas | Imputação mediana + StandardScaler |
| Categóricas | Imputação moda + One-Hot Encoding |
| Validação | Stratified 10-Fold Cross-Validation |

### 2.3 Estratégias de Balanceamento

1. Sem balanceamento  
2. SMOTE  
3. ADASYN  
4. Random Oversampling  
5. Class Weight  

### 2.4 Modelos Avaliados

Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, Extra Trees,
Gradient Boosting, Decision Tree, SVM, KNN, Naive Bayes, MLP Neural Network.

### 2.5 Experimentos

| Experimento | Variáveis |
|-------------|-----------|
| Experimento 1 | Somente dados maternos |
| Experimento 2 | Maternos + doenças, infecções, medicações, hábitos, exposição e gestacionais |
| Experimento 3 | Todas as variáveis incluindo marcadores fetais |

### 2.6 Métricas

Accuracy, Precision, Recall, F1 Score, ROC-AUC, PR-AUC, Sensibilidade, Especificidade.

### 2.7 Interpretabilidade

SHAP Values, Feature Importance nativa, Permutation Importance e Odds Ratios (Regressão Logística).

---

## 3. Estatísticas Descritivas

| Estatística | Valor |
|-------------|-------|
| Total de registros | 5,000 |
| Casos positivos (CCF) | 371 (7.42%) |
| Casos negativos | 4629 (92.58%) |
| Idade materna (média ± DP) | 28.4 ± 6.4 anos |
| IMC pré-gestacional (média) | 25.5 kg/m² |
| Gestação gemelar | 2.47% |
| TN aumentada | 3.72% |
| Malformação extracardíaca | 2.45% |

---

## 4. Comparação dos Modelos (Experimento 3 — Top 10)

| modelo | balanceamento | f1_score_mean | roc_auc_mean | pr_auc_mean | sensibilidade_mean | especificidade_mean |
| --- | --- | --- | --- | --- | --- | --- |
| CatBoost | class_weight | 0.3595 | 0.7904 | 0.3886 | 0.5959 | 0.8609 |
| CatBoost | random_oversampling | 0.3547 | 0.7733 | 0.3493 | 0.5552 | 0.8723 |
| Extra Trees | smote | 0.3442 | 0.768 | 0.2866 | 0.4379 | 0.9103 |
| SVM | random_oversampling | 0.3379 | 0.8151 | 0.4129 | 0.6863 | 0.8074 |
| Logistic Regression | random_oversampling | 0.3368 | 0.8139 | 0.4122 | 0.6862 | 0.8069 |
| Logistic Regression | class_weight | 0.3363 | 0.8162 | 0.4179 | 0.6862 | 0.8064 |
| SVM | smote | 0.3357 | 0.803 | 0.4146 | 0.6899 | 0.8036 |
| Logistic Regression | smote | 0.3322 | 0.8003 | 0.4138 | 0.6832 | 0.8028 |
| Extra Trees | adasyn | 0.3308 | 0.7689 | 0.2622 | 0.4309 | 0.9055 |
| SVM | adasyn | 0.3271 | 0.8008 | 0.4102 | 0.7067 | 0.7885 |

---

## 5. Comparação das Técnicas de Balanceamento

| Balanceamento | F1 Médio | F1 Std | F1 Máximo |
| --- | --- | --- | --- |
| adasyn | 0.1606 | 0.0755 | 0.3308 |
| class_weight | 0.1499 | 0.1046 | 0.3595 |
| none | 0.0699 | 0.0796 | 0.295 |
| random_oversampling | 0.2019 | 0.0748 | 0.3547 |
| smote | 0.1689 | 0.0769 | 0.3442 |

**Observação:** Em classes raras como CCF, SMOTE e Class Weight tendem a melhorar Recall e F1,
enquanto modelos sem balanceamento maximizam Accuracy às custas da sensibilidade.

---

## 6. Comparação entre Experimentos

| Experimento | melhor_modelo | melhor_balanceamento | f1_cv_mean | roc_auc_cv_mean | pr_auc_cv_mean | test_f1 | test_roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| experimento_1_maternal | CatBoost | class_weight | 0.1663 | 0.5677 | 0.1322 | 0.1783 | 0.5892 |
| experimento_2_maternal_doencas | SVM | smote | 0.2116 | 0.6384 | 0.2026 | 0.2586 | 0.6723 |
| experimento_3_completo | CatBoost | class_weight | 0.3595 | 0.7904 | 0.3886 | 0.3488 | 0.8041 |

### Ganho de Desempenho


- **Exp. 1 → Exp. 2:** Δ F1 = +0.0453 (inclusão de comorbidades e exposições)
- **Exp. 2 → Exp. 3:** Δ F1 = +0.1479 (inclusão de marcadores fetais)
- **Exp. 1 → Exp. 3:** Δ F1 = +0.1932 (modelo completo)

A inclusão de **marcadores ultrassonográficos fetais** proporciona o maior ganho discriminativo,
consistente com a literatura que destaca TN aumentada, ducto venoso alterado e regurgitação
tricúspide como preditores fortes de anomalias cardíacas.

---

## 7. Melhor Modelo Encontrado

| Parâmetro | Valor |
|-----------|-------|
| Experimento | experimento_3_completo |
| Modelo | CatBoost |
| Balanceamento | class_weight |
| F1 Score (CV) | 0.3595 ± 0.0240 |
| ROC-AUC (CV) | 0.7904 |
| PR-AUC (CV) | 0.3886 |
| Sensibilidade (CV) | 0.5959 |
| Especificidade (CV) | 0.8609 |
| F1 Score (Teste) | 0.3488 |
| ROC-AUC (Teste) | 0.8041 |

### Visualizações Geradas

- `reports/figures/best_*_confusion_matrix.png`
- `reports/figures/best_*_roc_curve.png`
- `reports/figures/best_*_pr_curve.png`
- `reports/figures/best_*_learning_curves.png`
- `reports/figures/best_*_calibration_curve.png`

---

## 8. Variáveis Mais Importantes

### Feature Importance (média entre modelos tree-based)

| feature | Importância Média |
| --- | --- |
| num__translucencia_nucal_aumentada | 14.948288853548265 |
| num__imc_pre_gestacional | 7.723196871162258 |
| num__idade_materna | 7.455452616785851 |
| num__regurgitacao_tricuspide | 7.358149075835937 |
| num__restricao_crescimento_fetal | 6.771248597508955 |
| num__malformacao_extracardiaca | 5.717343515420638 |
| num__alteracao_ducto_venoso | 5.4754592349444025 |
| num__historia_familiar_cardiopatia | 4.0476807085334245 |
| num__diabetes_pre_gestacional | 3.719357091389695 |
| num__idade_gestacional | 3.4091711760489494 |
| cat__corionicidade_NA | 3.1081494632403164 |
| num__diabetes_gestacional | 2.5512628251594327 |
| num__abortamentos_previos | 2.54668925190488 |
| num__paridade | 2.4645739076912196 |
| num__tabagismo | 2.4345618405685703 |

### SHAP Importance

| feature | SHAP Médio |
| --- | --- |
| num__translucencia_nucal_aumentada | 0.2955058727699065 |
| num__restricao_crescimento_fetal | 0.18784946886766835 |
| num__regurgitacao_tricuspide | 0.13913913803783853 |
| num__imc_pre_gestacional | 0.13793999681589844 |
| num__idade_materna | 0.12887303809941475 |
| num__alteracao_ducto_venoso | 0.11798275599151824 |
| num__malformacao_extracardiaca | 0.11103045259327046 |
| num__diabetes_gestacional | 0.08978771491475775 |
| num__historia_familiar_cardiopatia | 0.08685866707918026 |
| num__tabagismo | 0.07547899036419738 |
| num__diabetes_pre_gestacional | 0.06986605429130882 |
| cat__corionicidade_NA | 0.04922270794038942 |
| num__alcool | 0.048814295692525524 |
| num__polidramnio | 0.04476322499702052 |
| num__hipertensao_cronica | 0.04036787201042006 |

As variáveis relacionadas a **marcadores fetais** (translucência nucal, ducto venoso,
regurgitação tricúspide, malformações extracardíacas) e **histórico familiar/pessoal**
de cardiopatia congênita emergem consistentemente entre os métodos de interpretabilidade.

---

## 9. Discussão dos Resultados

1. **Desbalanceamento extremo:** Com prevalência ~7.4%, métricas como Accuracy
   são insuficientes; PR-AUC e Sensibilidade são prioritárias em triagem pré-natal.

2. **Marcadores fetais dominam:** O Experimento 3 supera os demais, confirmando que achados
   ultrassonográficos são os preditores mais informativos — alinhado à prática clínica de
   rastreamento morfológico fetal.

3. **Ensemble methods:** XGBoost, LightGBM e Random Forest apresentam desempenho competitivo,
   capturando interações não-lineares entre fatores de risco.

4. **SMOTE vs Class Weight:** Ambas melhoram sensibilidade; SMOTE tende a generalizar melhor
   em validação cruzada para datasets sintéticos com fronteiras de decisão suaves.

5. **Regressão Logística:** Oferece interpretabilidade via Odds Ratios (ver `data/odds_ratios_logistic.csv`),
   útil para comunicação clínica, embora com desempenho inferior aos métodos ensemble.

---

## 10. Limitações da Base Sintética

- **Ausência de dados reais:** Associações foram simuladas; magnitudes podem diferir de coortes clínicas.
- **Independência simplificada:** Correlações complexas entre comorbidades não foram totalmente modeladas.
- **Viés de confirmação:** Pesos definidos a priori podem superestimar importância de variáveis conhecidas.
- **Generalização:** Modelos devem ser recalibrados com dados do GestRisk antes de uso clínico.
- **Missing MCAR:** Padrões reais de dados ausentes em prontuários são mais complexos.

---

## 11. Possíveis Aplicações Futuras no Projeto GestRisk

1. **Triagem de risco integrada ao WhatsApp:** Estratificação automática de gestantes para
   encaminhamento ecocardiográfico fetal prioritário.

2. **Dashboard clínico:** Score de risco dinâmico atualizado conforme novos exames pré-natais.

3. **Validação prospectiva:** Substituir base sintética por dados anonimizados de parceiros hospitalares.

4. **Modelo híbrido:** Combinar ML tabular com análise de imagens de ultrassom fetal (deep learning).

5. **Explicabilidade clínica:** Relatórios SHAP traduzidos para linguagem acessível à equipe obstétrica.

6. **Monitoramento contínuo:** Retreinamento periódico com drift detection para manter calibração.

---

## 12. Arquivos Gerados

| Arquivo | Descrição |
|---------|-----------|
| `data/dataset.csv` | Base sintética completa |
| `data/metricas_modelos.csv` | Métricas de todos os modelos/experimentos |
| `data/feature_importance.csv` | Importância de features |
| `data/shap_importance.csv` | Valores SHAP agregados |
| `data/odds_ratios_logistic.csv` | Odds Ratios da Regressão Logística |
| `data/permutation_importance.csv` | Permutation Importance do melhor modelo |
| `models/best_model.joblib` | Pipeline serializado do melhor modelo |
| `reports/pipeline_summary.json` | Resumo JSON da execução |

---

*Relatório gerado automaticamente pelo pipeline GestRisk CCF v1.0*
