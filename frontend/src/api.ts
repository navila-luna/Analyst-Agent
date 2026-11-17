const API_BASE_URL = 'http://localhost:8000'

export interface Reference {
  index: number
  doc_name: string
  chunk_index: number
  similarity: number | null
}

export interface AskResponse {
  answer: string
  in_docs: boolean
  references: Reference[]
  searches_performed?: number
  tool_calls_performed?: number
  termination_reason?: string
}

export async function askQuestion(question: string): Promise<AskResponse> {
  const response = await fetch(`${API_BASE_URL}/analyst-agent/v1/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })

  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`)
  }

  return response.json()
}

export interface BotConfig {
  tone: string
  answer_format: string
  require_citations: boolean
}

export async function getConfig(): Promise<BotConfig> {
  const response = await fetch(`${API_BASE_URL}/config`)
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`)
  }
  return response.json()
}

export async function updateConfig(config: BotConfig): Promise<BotConfig> {
  const response = await fetch(`${API_BASE_URL}/config`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  })
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`)
  }
  return response.json()
}

export interface IndexResult {
  files_new: string[]
  files_changed: string[]
  files_unchanged: string[]
  files_deleted: string[]
  chunks_embedded: number
}

export async function indexFiles(files: FileList): Promise<IndexResult> {
  const formData = new FormData()
  for (const file of files) {
    formData.append('files', file)
  }

  const response = await fetch(`${API_BASE_URL}/index`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`)
  }

  return response.json()
}
