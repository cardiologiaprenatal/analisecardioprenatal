# Prompt — Exibir análise descritiva (CSV) no frontend Laravel

Cole este prompt em um chat Cursor aberto no repositório **sistemapretriagem** (Laravel 12).

---

## Contexto

O microsserviço Python **CardioPrenatal-ML** já está rodando em `http://127.0.0.1:8010` e expõe a análise estática do CSV (5.000 gestantes).

O Laravel **não** deve ler o CSV nem o MySQL do Python. Deve consumir apenas REST:

| Método | URL Python | Uso |
|--------|------------|-----|
| `GET` | `/graficos` | Catálogo JSON (`slug`, `titulo`, `disponivel`, `arquivo`, `url`) |
| `GET` | `/graficos/{slug}` | PNG do gráfico |
| `GET` | `/analise/descritiva` | Metadados: `total_gestantes`, `prevalencia_ccf`, `resumo_markdown`, lista de gráficos |

Configuração já esperada no `.env` Laravel:

```env
CCF_API_URL=http://host.docker.internal:8010
# ou local: http://127.0.0.1:8010
CCF_API_TIMEOUT=120
PYTHON_API_URL=http://host.docker.internal:8010
```

## O que já existe no Laravel (reutilizar)

- `App\Services\CcfApiClient`
  - `listarGraficosDescritivos()` → `GET /graficos`
  - `fetchGraficoDescritivo($slug)` → `GET /graficos/{slug}`
- Patch pronto: `storage/app/patches/app/Http/Controllers/GraficosController.php`
  - proxy PNG em rota tipo `/dashboard/graficos/{slug}`
  - fallback local em `storage/app/gestrisk/figuras`
- `DashboardAnaliseService::catalogoGraficosDescritivos()` com slugs/títulos
- Dashboard atual (`resources/views/dashboard.blade.php`) só mostra gráficos Chart.js do **MySQL** (`analises_historico`). **Falta** a seção dos PNGs populacionais da API Python.

## Tarefa

Implementar no frontend Laravel a exibição da **análise descritiva populacional** vinda da API Python.

### Backend (mínimo necessário)

1. Garantir que `GraficosController` esteja em `app/Http/Controllers/` (aplicar o patch se ainda não estiver).
2. Registrar rotas autenticadas (mesmo middleware do dashboard), por exemplo:
   - `GET /dashboard/graficos/{slug}` → `GraficosController@descritivo` (`name: dashboard.graficos.descritivo`)
3. Em `CcfApiClient`, adicionar método `obterAnaliseDescritiva(): array` chamando `GET /analise/descritiva` (timeout curto, fail-soft: retornar `[]` se API cair).
4. Em `DashboardController@index`, injetar:
   - `$graficosDescritivos` = lista da API (`listarGraficosDescritivos()`), filtrando `disponivel === true`
   - `$analiseDescritiva` = metadados (`total_gestantes`, `prevalencia_ccf`) — **não** enviar o markdown inteiro para a view se for pesado; opcional resumo curto

### Frontend (dashboard)

Na `resources/views/dashboard.blade.php`, criar uma seção **nova e distinta** da seção Chart.js atual:

**Título sugerido:** `Análise descritiva populacional (dataset / API ML)`

Conteúdo:

1. **KPIs** no topo da seção:
   - Total de gestantes (`total_gestantes`)
   - Prevalência CCF (`prevalencia_ccf` formatada em %)
   - Quantidade de gráficos disponíveis

2. **Grade de PNGs** (responsive):
   - Para cada item disponível, card com:
     - título (`titulo`)
     - `<img src="{{ route('dashboard.graficos.descritivo', $slug) }}" alt="...">`
   - Ordem sugerida dos slugs:
     `prevalencia`, `distribuicao`, `fatores_risco`, `risco`, `probabilidade`, `prioridade_eco`, `intercorrencias_cat`, `top_intercorrencias`, `importancia`, `experimentos`, `etnia`

3. **Estado vazio / erro**:
   - Se a API estiver offline ou `disponiveis === 0`, mostrar mensagem clara:
     > “Análise descritiva indisponível. Verifique se o microsserviço ML está em :8010 (`CCF_API_URL`).”

4. **Não confundir** com a seção existente “Gráficos — cohort analisado (MySQL)”. Manter as duas:
   - MySQL = pacientes reais do sistema
   - API ML = análise estática do CSV (cohort sintético/treino)

### UX / CSS

- Reutilizar classes do dashboard (`dash-section`, `main-card`, `dash-charts-grid`, etc.).
- PNGs com `max-width: 100%`, `height: auto`, borda/sombra leve coerente com o tema atual.
- Lazy-load: `loading="lazy"` nas imagens.
- Mobile: 1 coluna; desktop: 2 colunas.

### Segurança / arquitetura

- O browser **nunca** chama `127.0.0.1:8010` direto (CORS/Docker). Sempre via proxy Laravel (`/dashboard/graficos/{slug}`).
- Não persistir PNG no banco.
- Não alterar o contrato do microsserviço Python.
- Não conectar Laravel ao SQLite/arquivos do Python além do HTTP.

## Critérios de aceite

- [ ] Com a API Python no ar, o dashboard mostra a seção com KPIs + grade de imagens
- [ ] Cada imagem carrega via rota Laravel e retorna PNG 200
- [ ] Com a API offline, a seção mostra fallback amigável (sem quebrar o dashboard)
- [ ] A seção Chart.js (MySQL) continua funcionando
- [ ] Swagger Python permanece em `http://127.0.0.1:8010/docs` (não precisa linkar no UI, mas a integração HTTP deve usar `CCF_API_URL`)

## Como validar

1. Subir Python: `uvicorn app.main:app --host 0.0.0.0 --port 8010` (repo AnalisePython)
2. `curl http://127.0.0.1:8010/graficos` → 11 disponíveis
3. Subir Laravel (`php artisan serve` ou Docker na 8000)
4. Abrir `/dashboard` logado → ver PNGs
5. Abrir no browser `/dashboard/graficos/prevalencia` → imagem PNG

## Fora de escopo

- Não reimplementar `/predict` nesta tarefa
- Não migrar análise descritiva para Chart.js
- Não editar o repositório AnalisePython (já pronto)
