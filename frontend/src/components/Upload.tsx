import { useRef, useState } from 'react'
import { indexFiles, type IndexResult } from '../api'

export function Upload() {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [selectedNames, setSelectedNames] = useState<string[]>([])
  const [result, setResult] = useState<IndexResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleFilesSelected() {
    const files = fileInputRef.current?.files
    if (!files || files.length === 0) return

    setSelectedNames(Array.from(files).map((file) => file.name))
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const indexResult = await indexFiles(files)
      setResult(indexResult)
    } catch {
      setError('Something went wrong indexing those files. Is the backend running on port 8000?')
    } finally {
      setLoading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  return (
    <div className="upload">
      <p className="upload-title">Upload docs to index</p>

      <label htmlFor="doc-upload" className="upload-button">
        Choose files
      </label>
      <input
        id="doc-upload"
        ref={fileInputRef}
        type="file"
        multiple
        accept=".md,.txt"
        onChange={handleFilesSelected}
        disabled={loading}
        className="visually-hidden"
      />

      {selectedNames.length > 0 && <p className="selected-files">Selected: {selectedNames.join(', ')}</p>}
      {loading && <p>Indexing...</p>}
      {error && <p className="error">{error}</p>}

      {result && (
        <div className="index-result">
          <p>{result.chunks_embedded} chunk(s) embedded</p>
          {result.files_new.length > 0 && <p>New: {result.files_new.join(', ')}</p>}
          {result.files_changed.length > 0 && <p>Changed: {result.files_changed.join(', ')}</p>}
          {result.files_deleted.length > 0 && <p>Deleted: {result.files_deleted.join(', ')}</p>}
        </div>
      )}
    </div>
  )
}
