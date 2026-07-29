import { useEffect, useState } from 'react'
import { getConfig, updateConfig, type BotConfig } from '../api'
import { GearIcon } from '../icons'

export function Settings() {
  const [isOpen, setIsOpen] = useState(false)
  const [config, setConfig] = useState<BotConfig | null>(null)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    getConfig()
      .then((result) => {
        if (!cancelled) setConfig(result)
      })
      .catch(() => {
        if (!cancelled) setError('Could not load settings. Is the backend running on port 8000?')
      })

    return () => {
      cancelled = true
    }
  }, [])

  async function handleSave() {
    if (!config) return
    setSaving(true)
    setSaved(false)
    setError(null)

    try {
      const result = await updateConfig(config)
      setConfig(result)
      setSaved(true)
    } catch {
      setError('Could not save settings.')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="settings-wrapper">
      <button
        type="button"
        className="settings-toggle"
        onClick={() => setIsOpen((open) => !open)}
        aria-label="Settings"
      >
        <GearIcon />
      </button>

      {isOpen && (
        <div className="settings-panel">
          {error && !config && <p className="error">{error}</p>}

          {!config && !error && <p>Loading settings...</p>}

          {config && (
            <>
              <p className="settings-title">Bot settings</p>

              <label>
                Tone
                <select
                  value={config.tone}
                  onChange={(event) => setConfig({ ...config, tone: event.target.value })}
                >
                  <option value="professional">Professional</option>
                  <option value="casual">Casual</option>
                  <option value="formal">Formal</option>
                </select>
              </label>

              <label>
                Answer format
                <select
                  value={config.answer_format}
                  onChange={(event) => setConfig({ ...config, answer_format: event.target.value })}
                >
                  <option value="paragraph">Paragraph</option>
                  <option value="bullet points">Bullet points</option>
                </select>
              </label>

              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={config.require_citations}
                  onChange={(event) => setConfig({ ...config, require_citations: event.target.checked })}
                />
                Require citations
              </label>

              <button type="button" onClick={handleSave} disabled={saving}>
                {saving ? 'Saving...' : 'Save settings'}
              </button>

              {saved && <p className="saved-message">Saved.</p>}
              {error && <p className="error">{error}</p>}
            </>
          )}
        </div>
      )}
    </div>
  )
}
