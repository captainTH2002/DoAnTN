import { createChatBotMessage } from 'react-chatbot-kit'
import CustomHeader from './CustomHeader'

const config = {
  botName: 'Trợ Lý AI',
  initialMessages: [createChatBotMessage('Xin chào! Tôi có thể giúp gì về tài liệu của bạn?')],
  customComponents: {
    header: () => <CustomHeader />, // thay header
  },
  customStyles: {
    botMessageBox: {
      backgroundColor: '#563d7c',
      borderRadius: '15px',
      padding: '8px 12px',
      maxWidth: '80%',
    },
    chatButton: {
      backgroundColor: '#563d7c',
    },
    userMessageBox: {
      backgroundColor: '#4CAF50',
      color: '#fff',
      borderRadius: '15px',
      padding: '8px 12px',
      maxWidth: '80%',
    },
  },
}

export default config
