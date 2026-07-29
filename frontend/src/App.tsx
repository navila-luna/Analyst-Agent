import { Chat } from './components/Chat'
import { Upload } from './components/Upload'
import './App.css'

function App() {
  return (
    <main className="app">
      <h1>RAG Knowledge Bot</h1>
      <Upload />
      <hr />
      <Chat />
    </main>
  )
}

export default App
