"""
NEXUS Ω — Tool Interfaces v3.6.0

Contratos para herramientas futuras.
Estas clases son interfaces abstractas — NO implementan lógica real.
Sirven como blueprint para cuando se agreguen las herramientas.

Herramientas futuras planificadas:
  WebSearchTool       → búsqueda web en tiempo real
  DocumentReaderTool  → leer PDFs, DOCx, texto
  CalculatorTool      → cálculos matemáticos avanzados
  FileManagerTool     → leer archivos (solo lectura)
  SimulationTool      → simular escenarios
  KnowledgeSearchTool → búsqueda en base de conocimiento
  CodeAnalysisTool    → analizar código (sin ejecutar)
  DesignTool          → asistencia para diseño/arquitectura
"""

from __future__ import annotations

from backend.tools.base import BaseTool, RiskLevel, ToolInput, ToolResult


class WebSearchTool(BaseTool):
    """
    Búsqueda web en tiempo real.

    ESTADO: Interfaz definida. NO implementada.
    PREREQUISITO: API key de búsqueda (Tavily, SerpAPI, etc.)
    RIESGO: HIGH — llama a servicios externos.
    """
    enabled    = False
    risk_level = RiskLevel.HIGH

    @property
    def name(self) -> str: return "web_search"

    @property
    def description(self) -> str:
        return "Busca información actual en Internet."

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query":   {"type": "string"},
                "max_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        }

    @property
    def intent_keywords(self) -> list[str]:
        return ["busca", "buscar", "search", "web", "internet", "noticias", "actual"]

    async def execute(self, tool_input: ToolInput) -> ToolResult:
        return ToolResult(
            success=False,
            output=None,
            tool_name=self.name,
            error="WebSearchTool no implementada todavía.",
        )


class DocumentReaderTool(BaseTool):
    """
    Lee y extrae contenido de documentos (PDF, DOCX, TXT).

    ESTADO: Interfaz definida. NO implementada.
    RIESGO: LOW — solo lectura.
    """
    enabled    = False
    risk_level = RiskLevel.LOW

    @property
    def name(self) -> str: return "document_reader"

    @property
    def description(self) -> str:
        return "Lee y extrae texto de documentos PDF, DOCX o TXT."

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "pages":     {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["file_path"],
        }

    @property
    def intent_keywords(self) -> list[str]:
        return ["lee", "leer", "documento", "pdf", "archivo", "extrae"]

    async def execute(self, tool_input: ToolInput) -> ToolResult:
        return ToolResult(
            success=False, output=None, tool_name=self.name,
            error="DocumentReaderTool no implementada todavía.",
        )


class CalculatorTool(BaseTool):
    """
    Cálculos matemáticos seguros.

    ESTADO: Interfaz definida. NO implementada.
    NOTA: Implementar con sympy o mathjs — NO con eval().
    RIESGO: LOW.
    """
    enabled    = False
    risk_level = RiskLevel.LOW

    @property
    def name(self) -> str: return "calculator"

    @property
    def description(self) -> str:
        return "Realiza cálculos matemáticos, estadísticos y financieros."

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "expression": {"type": "string"},
                "variables":  {"type": "object"},
            },
            "required": ["expression"],
        }

    @property
    def intent_keywords(self) -> list[str]:
        return ["calcula", "calcular", "cuánto", "resultado", "formula", "matemática"]

    async def execute(self, tool_input: ToolInput) -> ToolResult:
        return ToolResult(
            success=False, output=None, tool_name=self.name,
            error="CalculatorTool no implementada todavía.",
        )


class KnowledgeSearchTool(BaseTool):
    """
    Búsqueda en base de conocimiento interna.

    ESTADO: Interfaz definida. Implementar con vector store.
    RIESGO: LOW.
    """
    enabled    = False
    risk_level = RiskLevel.LOW

    @property
    def name(self) -> str: return "knowledge_search"

    @property
    def description(self) -> str:
        return "Busca en la base de conocimiento interna de NEXUS."

    @property
    def intent_keywords(self) -> list[str]:
        return ["conocimiento", "base de datos", "información interna"]

    async def execute(self, tool_input: ToolInput) -> ToolResult:
        return ToolResult(
            success=False, output=None, tool_name=self.name,
            error="KnowledgeSearchTool no implementada todavía.",
        )


class CodeAnalysisTool(BaseTool):
    """
    Analiza código sin ejecutarlo.

    ESTADO: Interfaz definida. NO implementada.
    IMPORTANTE: Solo análisis estático. NUNCA exec().
    RIESGO: LOW (solo lectura).
    """
    enabled    = False
    risk_level = RiskLevel.LOW

    @property
    def name(self) -> str: return "code_analysis"

    @property
    def description(self) -> str:
        return "Analiza código: complejidad, errores, estilo. Sin ejecutar."

    @property
    def intent_keywords(self) -> list[str]:
        return ["analiza código", "revisa código", "bug", "código", "función"]

    async def execute(self, tool_input: ToolInput) -> ToolResult:
        return ToolResult(
            success=False, output=None, tool_name=self.name,
            error="CodeAnalysisTool no implementada todavía.",
        )


# Herramientas explícitamente prohibidas — nunca implementar sin sandbox seguro
FORBIDDEN_TOOLS = [
    "shell_exec",       # exec() / system()
    "file_write",       # escritura irrestricta
    "file_delete",      # eliminación
    "network_raw",      # conexiones arbitrarias
    "process_spawn",    # spawn procesos
    "system_config",    # modificar configuración del sistema
]


def register_future_tools(registry) -> None:
    """
    Registra las interfaces de tools futuras en el registry.
    Todas estarán deshabilitadas hasta su implementación.
    """
    future_tools = [
        WebSearchTool(),
        DocumentReaderTool(),
        CalculatorTool(),
        KnowledgeSearchTool(),
        CodeAnalysisTool(),
    ]
    for tool in future_tools:
        registry.register(tool)
