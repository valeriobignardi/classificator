export interface TrainingFileInfo {
  name: string;
  path: string;
  size?: number;
  modified_at?: string;
}

export interface TrainingFileListResponse {
  success: boolean;
  tenant_id: string;
  tenant_name: string;
  tenant_slug: string;
  files: TrainingFileInfo[];
}

export interface TrainingFileContentResponse {
  success: boolean;
  file: { name: string; path: string };
  content: string;
  truncated: boolean;
  limit: number;
  total_lines: number;
}

