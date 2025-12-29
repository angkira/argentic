export type NodeType = 'agent' | 'supervisor' | 'tool' | 'input' | 'output';

export interface NodePosition {
  x: number;
  y: number;
}

export interface WorkflowNodeData {
  label: string;
  config: Record<string, any>;
}

export interface WorkflowNode {
  id: string;
  type: NodeType;
  position: NodePosition;
  data: WorkflowNodeData;
}

export interface WorkflowEdge {
  id: string;
  source: string;
  target: string;
  label?: string;
  type?: string;
}

export interface WorkflowConfig {
  name: string;
  description: string;
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
}

export interface WorkflowCreate extends WorkflowConfig {}

export interface WorkflowUpdate {
  name?: string;
  description?: string;
  nodes?: WorkflowNode[];
  edges?: WorkflowEdge[];
}

export interface Workflow extends WorkflowConfig {
  id: string;
  created_at: string;
  updated_at: string;
  status: string;
}
