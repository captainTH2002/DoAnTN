import React, { useEffect, useState, useRef } from 'react'
import axios from '../../api/axios'

const ChatbotPage = () => {
  const [conversations, setConversations] = useState([])
  const [selectedId, setSelectedId] = useState(null)
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const messageEndRef = useRef(null)

  useEffect(() => {
    loadConversations()
  }, [])
  useEffect(() => {
    if (selectedId) loadMessages(selectedId)
  }, [selectedId])

  useEffect(() => {
    messageEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const loadConversations = async () => {
    const res = await axios.get('/conversations')
    setConversations(res.data)
  }

  const loadMessages = async (id) => {
    const res = await axios.get(`/messages/${id}`)
    setMessages(res.data)
  }

  const createNewConversation = async () => {
    const res = await axios.post('/conversations', { userId: localStorage.getItem('userId') })
    setSelectedId(res.data._id)
    loadConversations()
    setMessages([])
  }

  const sendMessage = async () => {
    if (!input.trim()) return
    const userMsg = { role: 'user', message: input }
    setMessages((prev) => [...prev, userMsg])
    setInput('')

    const res = await axios.post('/chat', { conversationId: selectedId, question: input })
    setMessages((prev) => [...prev, { role: 'bot', message: res.data.answer }])
  }

  return (
    <div className="flex h-[90vh] bg-gray-900 text-white">
      {/* Sidebar */}
      <div className="w-64 bg-gray-800 border-r border-gray-700 p-4 overflow-y-auto">
        <button
          onClick={createNewConversation}
          className="w-full mb-4 py-2 bg-purple-600 rounded hover:bg-purple-700 transition"
        >
          + New Chat
        </button>
        <ul>
          {conversations.map((conv) => (
            <li
              key={conv._id}
              className={`p-2 rounded cursor-pointer mb-2 text-sm hover:bg-purple-700/30 ${
                selectedId === conv._id ? 'bg-purple-700/50 font-semibold' : ''
              }`}
              onClick={() => setSelectedId(conv._id)}
            >
              {conv.title}
            </li>
          ))}
        </ul>
      </div>

      {/* Chat */}
      <div className="flex-1 flex flex-col">
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`px-5 py-3 rounded-2xl max-w-[70%] whitespace-pre-wrap 
                  shadow transition 
                  ${
                    msg.role === 'user'
                      ? 'bg-gradient-to-br from-green-500 to-green-600 text-white'
                      : 'bg-gray-700 text-white'
                  }`}
              >
                {msg.message}
              </div>
            </div>
          ))}
          <div ref={messageEndRef} />
        </div>

        <div className="p-4 border-t border-gray-700 flex items-center gap-2 bg-gray-800">
          <input
            className="flex-1 bg-gray-700 border border-gray-600 rounded-full px-4 py-2 outline-none focus:ring-2 focus:ring-purple-500 text-white"
            placeholder="Type your message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
          />
          <button
            onClick={sendMessage}
            className="bg-green-500 px-5 py-2 rounded-full hover:bg-green-600 transition"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  )
}

export default ChatbotPage
