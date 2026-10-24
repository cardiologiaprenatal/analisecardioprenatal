# API Key — integração Laravel ↔ CardioPrenatal-ML

## Python (já implementado)

1. Defina no `.env` do microsserviço:

```env
API_KEY=sua_chave_forte_aqui
```

2. Rotas protegidas exigem um destes headers:
   - `X-API-Key: sua_chave_forte_aqui`
   - `Authorization: Bearer sua_chave_forte_aqui`

3. Rotas públicas (sem chave): `/`, `/health`, `/docs`, `/openapi.json`, `/redoc`

## Laravel — o que configurar

### 1) `.env`

```env
PYTHON_API_URL=http://127.0.0.1:8010
CCF_API_URL=http://127.0.0.1:8010
PYTHON_API_KEY=sua_chave_forte_aqui
CCF_API_KEY="${PYTHON_API_KEY}"
```

Use **exatamente a mesma** chave do `API_KEY` do Python.

### 2) `config/services.php` (exemplo)

```php
'python_ml' => [
    'url' => env('PYTHON_API_URL', 'http://127.0.0.1:8010'),
    'key' => env('PYTHON_API_KEY'),
],
'ccf' => [
    'url' => env('CCF_API_URL', env('PYTHON_API_URL', 'http://127.0.0.1:8010')),
    'key' => env('CCF_API_KEY', env('PYTHON_API_KEY')),
],
```

### 3) Cliente HTTP — enviar o header em todas as chamadas

Em `PythonApiService` / `CcfApiClient`, no método base do HTTP:

```php
protected function client(): \Illuminate\Http\Client\PendingRequest
{
    $pending = Http::baseUrl(rtrim(config('services.python_ml.url'), '/'))
        ->acceptJson()
        ->timeout(30);

    $key = config('services.python_ml.key');
    if (!empty($key)) {
        $pending = $pending->withHeaders(['X-API-Key' => $key]);
    }

    return $pending;
}
```

Para o `CcfApiClient`, o mesmo padrão com `config('services.ccf.url')` e `config('services.ccf.key')`.

### 4) Teste rápido

```bash
# sem chave → 401
curl -s -o - -w "%{http_code}" http://127.0.0.1:8010/gestantes

# com chave → 200
curl -s -H "X-API-Key: sua_chave_forte_aqui" http://127.0.0.1:8010/health
curl -s -H "X-API-Key: sua_chave_forte_aqui" "http://127.0.0.1:8010/gestantes?limit=1"
```
