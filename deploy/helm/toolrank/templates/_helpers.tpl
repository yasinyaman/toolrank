{{- define "toolrank.fullname" -}}
{{- if contains .Chart.Name .Release.Name -}}{{ .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else -}}{{ printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" }}{{- end -}}
{{- end -}}

{{- define "toolrank.labels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
{{- end -}}

{{- define "toolrank.selector" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "toolrank.image" -}}
{{- if eq .Values.embedding.mode "bundled" -}}
{{ required "embedding.bundled.image: build deploy/docker/Dockerfile.vllm and push it (the image is not published)" .Values.embedding.bundled.image }}
{{- else -}}
{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}
{{- end -}}
{{- end -}}

{{/* the served model name: the external one, or the profile's */}}
{{- define "toolrank.model" -}}
{{- if and (eq .Values.embedding.mode "external") .Values.embedding.model -}}{{ .Values.embedding.model }}
{{- else if eq .Values.embedding.profile "fp8" -}}{{ .Values.embedding.backbone.name }}-fp8
{{- else if eq .Values.embedding.profile "bf16" -}}{{ .Values.embedding.backbone.name }}
{{- else -}}{{ fail (printf "embedding.profile: fp8 or bf16, not %q" .Values.embedding.profile) }}
{{- end -}}
{{- end -}}

{{- define "toolrank.embUrl" -}}
{{- if eq .Values.embedding.mode "vllm" -}}http://{{ include "toolrank.fullname" . }}-embedding:8000/v1
{{- else if eq .Values.embedding.mode "external" -}}{{ required "embedding.url: the /v1 endpoint of your embedding server" .Values.embedding.url }}
{{- else if eq .Values.embedding.mode "bundled" -}}http://127.0.0.1:8091/v1
{{- else -}}{{ fail (printf "embedding.mode: vllm, external or bundled, not %q" .Values.embedding.mode) }}
{{- end -}}
{{- end -}}

{{- define "toolrank.authSecret" -}}{{ printf "%s-auth" (include "toolrank.fullname" .) }}{{- end -}}

{{- define "toolrank.checkAuth" -}}
{{- if not (or .Values.auth.apiKey .Values.auth.apiKeySecret.name .Values.auth.apiKeys .Values.auth.apiKeysSecret.name) -}}
{{- fail "auth: set auth.apiKey, auth.apiKeySecret, auth.apiKeys or auth.apiKeysSecret (the server is reachable on the pod network)" -}}
{{- end -}}
{{- end -}}
