import { Chat } from './components/Chat'
import { Settings } from './components/Settings'
import { Upload } from './components/Upload'
import './App.css'

function App() {
  return (
    <main className="app">
      <div className="app-header">
        <h1>RAG Knowledge Bot</h1>
        <Settings />
      </div>
      <hr />
      <Upload />
      <hr />
      <Chat />
    </main>
  )
}

export default App
