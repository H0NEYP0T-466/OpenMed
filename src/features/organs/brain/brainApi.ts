import type { BrainClassificationResult, BrainModelInfo } from './brainTypes';

const API_BASE_URL = 'http://localhost:8016/api/brain';

export const classifyBrainImage = async (file: File): Promise<BrainClassificationResult> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/classify`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    if (response.status === 503) {
      throw new Error('Model is currently loading or unavailable (503). Please try again in a moment.');
    }
    const errorData = await response.json().catch(() => null);
    throw new Error(errorData?.detail || `Classification failed with status ${response.status}`);
  }

  return response.json();
};

export const getBrainModelInfo = async (): Promise<BrainModelInfo> => {
  const response = await fetch(`${API_BASE_URL}/model-info`);
  
  if (!response.ok) {
    throw new Error(`Failed to fetch model info (Status: ${response.status})`);
  }

  return response.json();
};

export const getBrainHealth = async (): Promise<{ status: string }> => {
  const response = await fetch(`${API_BASE_URL}/health`);
  
  if (!response.ok) {
    throw new Error(`Health check failed (Status: ${response.status})`);
  }

  return response.json();
};
