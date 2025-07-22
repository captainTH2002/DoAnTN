import axios from '../../api/axios'

class ActionProvider {
  constructor(createChatBotMessage, setStateFunc) {
    this.createChatBotMessage = createChatBotMessage
    this.setState = setStateFunc
  }

  async handleQuestion(message) {
    try {
      const res = await axios.post('/chat', { question: message })
      console.log('== Contexts from server:', res.data.contexts)
      const botMessage = this.createChatBotMessage(res.data.answer)
      this.setState((prev) => ({
        ...prev,
        messages: [...prev.messages, botMessage],
      }))
    } catch (error) {
      const botMessage = this.createChatBotMessage('Xin lỗi, hiện tại có lỗi.')
      this.setState((prev) => ({
        ...prev,
        messages: [...prev.messages, botMessage],
      }))
    }
  }
}

export default ActionProvider

// import axios from '../../api/axios'

// class ActionProvider {
//   constructor(createChatBotMessage, setStateFunc, conversationId) {
//     this.createChatBotMessage = createChatBotMessage
//     this.setState = setStateFunc
//     this.conversationId = conversationId
//   }

//   async handleQuestion(message, conversationId) {
//     try {
//       // Lưu message user
//       await axios.post('/messages', {
//         conversationId,
//         role: 'user',
//         message,
//       })

//       // Gửi câu hỏi đến AI
//       const res = await axios.post('/chat', { question: message })
//       console.log('== Server trả:', res.data)

//       const answer = res.data.answer

//       // Lưu message bot
//       await axios.post('/messages', {
//         conversationId,
//         role: 'bot',
//         message: answer,
//       })

//       // Hiển thị trên UI
//       const botMessage = this.createChatBotMessage(answer)
//       this.setState((prev) => ({
//         ...prev,
//         messages: [...prev.messages, botMessage],
//       }))
//     } catch (error) {
//       console.error('Lỗi:', error)
//       const botMessage = this.createChatBotMessage('Xin lỗi, hiện tại có lỗi.')
//       this.setState((prev) => ({
//         ...prev,
//         messages: [...prev.messages, botMessage],
//       }))
//     }
//   }
// }

// export default ActionProvider
