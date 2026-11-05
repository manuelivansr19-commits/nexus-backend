# NEXUS — Fase 2: acceso y secretos (runbook y variables)

Estado: implementado en una rama local de trabajo; **no desplegado**. Un solo propietario, tokens Bearer
por entorno/proyecto, sesión opaca en memoria, filtro de secretos. Fase 3 (persistencia por proyecto) **no** está incluida.

## 1. Qué protege y cómo
- **Tres capas:** (1) token Bearer, guardado solo como hash SHA-256 y comparado en tiempo constante;
  (2) propietario: contraseña con scrypt + cookie de sesión opaca (`__Host-nexus_session`; solo el hash del id vive en el servidor);
  (3) aislamiento por proyecto.
- **Orden:** si hay cabecera `Authorization`, debe ser un Bearer válido (si no, 401 y **no** se mira la cookie); si no, la cookie; si no, 401.
- **Clases de ruta** (tabla única en `backend/security/routes.py`; lo no declarado se niega): PÚBLICA 6, SESIÓN 2, UNLOCK 1,
  SISTEMA 6, PROYECTO 2, GLOBAL 17 (34 rutas; 28 protegidas). Un token de proyecto recibe `403 {"error":"global_route_denied"}`
  en las rutas GLOBAL **antes** de ejecutar el manejador. Solo puede usar las 6 rutas SISTEMA con `read` y la memoria persistente de SU proyecto.
- Las 5 rutas `/api/auth/*` las atiende el propio middleware (no aparecen en `app.routes`; sí en la tabla de rutas).
- **Encuadre HTTP:** el middleware lee el cuerpo completo (máx. `NEXUS_MAX_BODY_BYTES`, 1 MB) con plazo total (`NEXUS_BODY_TIMEOUT_SECONDS`)
  antes de entregarlo a FastAPI; toda respuesta generada dentro de ASGI a una petición con cuerpo lleva `Connection: close`.
  Lo que rechaza uvicorn/h11 antes de ASGI (p. ej. Content-Length inválido) **no** lo controla NEXUS.
- **Filtro de secretos** (heurístico, sin garantía de detectar todo): rechaza en guardados explícitos y argumentos de herramientas,
  redacta en guardados implícitos y en todo lo que sale hacia un runtime (local o de nube).

## 2. Variables de entorno
| Variable | Por defecto | Notas |
|---|---|---|
| NEXUS_ENVIRONMENT | production | production / development / test. `audit` y `off` solo se honran en development/test |
| NEXUS_AUTH_MODE | enforce | enforce / audit / off (ignorado en production) |
| NEXUS_OWNER_PASSWORD_HASH | — | `scrypt$N$r$p$sal$hash` (hex). Generar con `python -m backend.security.cli hash-password` |
| NEXUS_API_TOKENS | vacío | lista JSON; `projects` = `["*"]` o nombres, nunca mezclados |
| NEXUS_PUBLIC_ORIGIN | — | **Obligatoria para el login por navegador en todos los entornos.** Origen exacto sin barra final, https (http solo 127.0.0.1/localhost fuera de production) |
| NEXUS_DEFAULT_PROJECT | sin valor | sin ella, una petición sin `project` en rutas de proyecto da 400 `project_required` |
| NEXUS_SINGLE_PROCESS_ACK | false | **declaración del propietario**, ver §3 |
| NEXUS_ALLOWED_ORIGINS | vacío | orígenes exactos (coma) para CORS de clientes Bearer; nunca `*`, nunca credenciales |
| NEXUS_COOKIE_SECURE | true | `false` solo en development/test (la cookie pierde el prefijo `__Host-`) |
| NEXUS_SESSION_MAX_SECONDS / _IDLE_SECONDS / _MAX_ACTIVE | 28800 / 1800 / 5 | 8 h, 30 min, sesiones simultáneas |
| NEXUS_LOGIN_MAX_ATTEMPTS, NEXUS_LOGIN_WINDOW_SECONDS, NEXUS_LOGIN_LOCKOUT_SECONDS, NEXUS_LOGIN_BACKOFF_SECONDS | 5 / 900 / 900 / 1 | retardo 1, 2, 4, 8 s; quinto fallo = bloqueo 900 s |
| NEXUS_RATE_CHAT_PER_MINUTE / _TASK_ / NEXUS_UNLOCK_RATE_PER_MINUTE | 30 / 5 / 5 | por principal y por proceso |
| NEXUS_MAX_BODY_BYTES / NEXUS_BODY_TIMEOUT_SECONDS | 1048576 / 30 | |

Valores inválidos: se usa el defecto con una advertencia que nombra la variable (nunca el valor). JSON de tokens inválido desactiva TODOS los tokens.

## 3. NEXUS_SINGLE_PROCESS_ACK (texto normativo)
La Fase 2 solo se soporta con **un proceso y una instancia**. `NEXUS_SINGLE_PROCESS_ACK=true` es una **declaración del propietario**:
NEXUS **no puede verificarla**: NEXUS no detecta las instancias de Render (ni el comando de arranque ni el escalado). Solo habilita el login por navegador;
sin ella los Bearer siguen funcionando.
1. La verificación es **manual** y tuya, en el panel de Render: número de instancias, comando de arranque (sin `--workers` mayor que 1), `WEB_CONCURRENCY`. Es una foto de un momento.
2. `boot_id` (en `/api/nexus/status`, con `read`) y la guardia que lee `WEB_CONCURRENCY`/`UVICORN_WORKERS` son **señales, no pruebas**.
3. **Cambio de topología** (instancias, plan, escalado, comando de arranque, WEB_CONCURRENCY/UVICORN_WORKERS): el login por navegador debe quedar
   **deshabilitado** hasta repetir la verificación: pon `NEXUS_SINGLE_PROCESS_ACK=false` (o elimínala) y reinicia; vuelve a `true` solo tras verificar.
4. Limitación declarada: NEXUS no detecta por sí mismo que la topología cambió; si cambia sin que lo notes, el login seguirá habilitado con 401 intermitentes y límites multiplicados por N.
5. El registro de arranque con ACK dice «declaración del propietario, no verificada por NEXUS» y nunca afirma que haya una sola instancia.

## 4. Runbook de recuperación
- **R1 bloqueo temporal del propietario:** esperar el bloqueo, reiniciar el proceso, o `POST /api/auth/unlock` con un **Bearer admin con `["*"]`**
  (una cookie NO sirve). Sin Bearer admin solo queda esperar o reiniciar. `unlock` solo borra el contador de login y deja una línea de registro (id del token, hora, número).
- **R2 contraseña olvidada:** `python -m backend.security.cli hash-password` en tu equipo (sin red, sin eco) → cambiar `NEXUS_OWNER_PASSWORD_HASH` en Render → redesplegar. Cierra todas las sesiones.
- **R3 token comprometido:** `new-token`, añadir la entrada nueva, marcar la vieja `"revoked": true`, redesplegar (el viejo vale hasta ese reinicio).
- **R4 dispositivo perdido:** `logout-all` desde otra sesión, o reiniciar.
- **R5 sin ninguna credencial:** en local, `NEXUS_ENVIRONMENT=development`; en producción no hay atajo: se repite R2/R3 con el panel. El servicio responde 503, nunca abierto.
- **R6 — pérdida simultánea del Bearer admin y del acceso al panel de Render: NO existe recuperación local garantizada.** Está diseñado así a propósito
  (sin puerta trasera, sin ruta para cambiar la contraseña). Hay que recuperar la cuenta de Render por sus propios medios; los datos ni se pierden ni se exponen.
  Prevención: guarda el Bearer admin, la contraseña y las variables en tu gestor con acceso de emergencia.

## 5. Herramienta local
`python -m backend.security.cli new-token | hash-password | check-env | scan-stored` — sin red, sin archivos de salida, sin secretos en argumentos.

## 6. Activación segura (resumen; nada se hace sin autorización de despliegue)
1. En tu equipo: `hash-password`, `new-token`, anotar la URL https exacta del servicio, `check-env` (debe decir OK).
2. En Render, solo lectura: instancias, comando de arranque, `WEB_CONCURRENCY`, versión de uvicorn.
3. Cargar las variables **antes** del código; `NEXUS_SINGLE_PROCESS_ACK=true` solo tras el paso 2.
4. Desplegar y comprobar: 401 sin credencial, 403 `global_route_denied` con token de proyecto, login por navegador, cierre de sesión, `/health` 200.
5. Revisión manual del navegador (service workers / cachés viejos, cookies, «Atrás» tras cerrar sesión).

## 7. Límites conocidos
Sesiones, contadores y límites son por proceso. Revocar un token exige reinicio. El filtro de secretos es heurístico.
El proxy de Render no está verificado (encuadre entre proxy y uvicorn, temporizadores). Las pruebas de sockets describen la versión de uvicorn/h11 instalada donde se ejecutan.
