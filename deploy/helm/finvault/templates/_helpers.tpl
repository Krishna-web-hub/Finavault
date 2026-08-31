{{/*
Shared template helpers.
*/}}

{{- define "finvault.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "finvault.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{- define "finvault.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "finvault.labels" -}}
helm.sh/chart: {{ include "finvault.chart" . }}
{{ include "finvault.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: finvault
{{- end }}

{{/*
Selector labels are a subset of the full label set on purpose: a Deployment's
selector is immutable after creation, so anything that changes between
releases — the chart version, the app version — must stay out of it, or the
next `helm upgrade` fails with "field is immutable".
*/}}
{{- define "finvault.selectorLabels" -}}
app.kubernetes.io/name: {{ include "finvault.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "finvault.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "finvault.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
The Secret the pods read from — either one supplied out of band or the one
this chart creates.
*/}}
{{- define "finvault.secretName" -}}
{{- if .Values.secrets.existingSecret }}
{{- .Values.secrets.existingSecret }}
{{- else }}
{{- printf "%s-secrets" (include "finvault.fullname" .) }}
{{- end }}
{{- end }}

{{/*
Fail rendering — not deployment — when no secret source is configured.

`helm install` stopping with a readable message beats a Deployment that rolls
out, starts, and signs tokens with the empty string. The check runs at
template time, so it also fires in `helm template` and `helm lint` in CI,
before anything reaches a cluster.
*/}}
{{- define "finvault.validateSecrets" -}}
{{- if and (not .Values.secrets.existingSecret) (not .Values.secrets.create) }}
{{- fail "\n\nFinVault requires secrets. Either:\n  (a) set secrets.existingSecret to the name of a Secret containing FINVAULT_JWT_SECRET, LLM_API_KEY, POSTGRES_PASSWORD and FINVAULT_MASTER_KEY; or\n  (b) set secrets.create=true and provide secrets.jwtSecret, secrets.llmApiKey, secrets.postgresPassword and secrets.masterKey.\n\nThere is deliberately no default: a shipped default JWT secret is a shipped vulnerability.\n" }}
{{- end }}
{{- if and .Values.secrets.create (not .Values.secrets.jwtSecret) }}
{{- fail "secrets.create is true but secrets.jwtSecret is empty. Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(48))\"" }}
{{- end }}
{{- if and .Values.secrets.create (not .Values.secrets.masterKey) }}
{{- fail "secrets.create is true but secrets.masterKey is empty. This is the key that wraps every per-chunk data key — losing it makes every stored document permanently unreadable, so generate it deliberately and back it up before the first ingest." }}
{{- end }}
{{- end }}

{{/*
DSN assembled from parts so the password comes from the Secret at runtime
rather than being baked into a ConfigMap. `$(POSTGRES_PASSWORD)` is expanded
by the kubelet from the env var below it, not by Helm.
*/}}
{{- define "finvault.postgresDsn" -}}
postgresql://{{ .Values.postgres.user }}:$(POSTGRES_PASSWORD)@{{ .Values.postgres.host }}:{{ .Values.postgres.port }}/{{ .Values.postgres.database }}
{{- end }}
