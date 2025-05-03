import json
from typing import Dict, Any, Optional, Callable
from pydantic import BaseModel

from core.messager import Messager


class ToolManager:
    """Manages tool registration, grammar generation, and execution."""

    def __init__(self, messager: Messager):
        self.messager = messager
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.messager.log("ToolManager initialized.")

    def register_tool(
        self,
        tool_id: str,
        tool_name: str,
        tool_manual: str,
        argument_schema: Optional[type[BaseModel]],
        implementation: Callable,
    ) -> None:
        if tool_id in self.tools:
            self.messager.log(
                f"Warning: Tool with ID '{tool_id}' already registered. Overwriting.",
                level="warning",
            )

        self.tools[tool_id] = {
            "name": tool_name,
            "manual": tool_manual,
            "schema": argument_schema,
            "implementation": implementation,
        }
        self.messager.log(f"ToolManager registered tool '{tool_name}' (ID: {tool_id}).")

    def generate_tool_descriptions_for_prompt(self) -> str:
        if not self.tools:
            return "No tools are currently available."

        descriptions = []
        for tool_id, tool_data in self.tools.items():
            desc = f"- {tool_data['name']} (ID: {tool_id}): {tool_data['manual']}"
            schema = tool_data.get("schema")
            if schema:
                try:
                    schema_json = schema.schema()
                    properties = schema_json.get("properties", {})
                    required = schema_json.get("required", [])
                    args_desc = {}
                    for name, prop_schema in properties.items():
                        prop_desc = prop_schema.get("description", "")
                        prop_type = prop_schema.get("type", "any")
                        is_required = name in required
                        args_desc[name] = (
                            f"({prop_type}{', required' if is_required else ''}): {prop_desc}"
                        )
                    if args_desc:
                        desc += f"\\n  Arguments: {json.dumps(args_desc)}"
                except Exception as e:
                    self.messager.log(
                        f"Error generating schema description for tool {tool_id}: {e}",
                        level="warning",
                    )
            descriptions.append(desc)

        return "\\n".join(descriptions)

    def generate_tool_grammar(self) -> Optional[str]:
        if not self.tools:
            return None

        grammar_parts = [
            "root ::= response_json | plain_text",
            "plain_text ::= [^\\{]+",
            "ws ::= [ \\t\\n]*",
            'string ::= "\\" ([^"\\\\] | "\\\\" [bfnrt"\\\\/]) * "\\"',
            'number ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?',
            'boolean ::= "true" | "false"',
            'null ::= "null"',
            'response_json ::= "{" ws "\\"tool_calls\\"" ws ":" ws tool_call_list ws "}" ws',
            'tool_call_list ::= "[" ws (tool_call ws ("," ws tool_call)*)? "]" ws',
            'tool_call ::= "{" ws "\\"tool_id\\"" ws ":" ws string ws "," ws "\\"arguments\\"" ws ":" ws args_object ws "}" ws',
        ]

        arg_rules = []
        arg_object_options = []

        for tool_id, tool_data in self.tools.items():
            schema = tool_data.get("schema")
            rule_name = f"{tool_id}_args"
            arg_object_options.append(rule_name)

            if schema:
                try:
                    schema_json = schema.schema()
                    properties = schema_json.get("properties", {})
                    prop_rules = []
                    for i, (name, prop_schema) in enumerate(properties.items()):
                        prop_type = prop_schema.get("type", "string")
                        gbnf_type = "string"
                        if prop_type == "integer" or prop_type == "number":
                            gbnf_type = "number"
                        elif prop_type == "boolean":
                            gbnf_type = "boolean"
                        comma = '," ws' if i > 0 else ""
                        prop_rules.append(f'{comma} "\\"{name}\\"" ws ":" ws {gbnf_type}')
                    arg_rules.append(f'{rule_name} ::= "{{" ws { "".join(prop_rules) } ws "}}"')
                except Exception as e:
                    self.messager.log(
                        f"Error generating GBNF for tool {tool_id} schema: {e}", level="warning"
                    )
                    arg_rules.append(f"{rule_name} ::= json_object # Fallback for {tool_id}")
                    if (
                        'json_object ::= "{" ws ( string ws ":" ws json_value ws ("," ws string ws ":" ws json_value)* )? "}" ws'
                        not in grammar_parts
                    ):
                        grammar_parts.extend(
                            [
                                'json_object ::= "{" ws ( string ws ":" ws json_value ws ("," ws string ws ":" ws json_value)* )? "}" ws',
                                'json_array ::= "[" ws ( json_value ws ("," ws json_value)* )? "]" ws',
                                "json_value ::= string | number | boolean | null | json_object | json_array",
                            ]
                        )
            else:
                arg_rules.append(f'{rule_name} ::= "{{" ws "}}"')

        if not arg_object_options:
            return None

        grammar_parts.append(f'args_object ::= {" | ".join(arg_object_options)}')
        grammar_parts.extend(arg_rules)
        full_grammar = "\\n".join(grammar_parts)
        self.messager.log(f"Generated GBNF Grammar (tool_calls list format):\\n{full_grammar}")
        return full_grammar

    def execute_tool(self, tool_id: str, arguments: Dict[str, Any]) -> Any:
        if tool_id not in self.tools:
            return f"Error: Tool with ID '{tool_id}' not found."

        tool_data = self.tools[tool_id]
        implementation = tool_data.get("implementation")
        schema = tool_data.get("schema")

        if not implementation:
            return f"Error: Implementation missing for tool '{tool_id}'."

        try:
            if schema:
                validated_args = schema(**arguments)
                args_dict = validated_args.dict()
            else:
                if arguments:
                    return f"Error: Tool '{tool_id}' takes no arguments, but received {arguments}"
                args_dict = {}

            self.messager.log(f"Executing tool '{tool_id}' with arguments: {args_dict}")
            result = implementation(**args_dict)
            self.messager.log(f"Tool '{tool_id}' executed successfully. Result: {result}")
            return result
        except Exception as e:
            self.messager.log(f"Error executing tool '{tool_id}': {e}", level="error")
            return f"Error executing tool '{tool_id}': {e}"
