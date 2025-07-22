class MessageParser {
  constructor(actionProvider) {
    this.actionProvider = actionProvider
  }

  parse(message) {
    console.log('User gửi:', message)
    this.actionProvider.handleQuestion(message)
  }
}

export default MessageParser

// class MessageParser {
//   constructor(actionProvider, conversationId) {
//     this.actionProvider = actionProvider
//     this.conversationId = conversationId
//   }

//   parse(message) {
//     console.log('User gửi:', message)
//     this.actionProvider.handleQuestion(message, this.conversationId)
//   }
// }

// export default MessageParser
