use serde_json::json;

use crate::normalized::{NormalizedBlock, NormalizedMessage, NormalizedRole};

pub(crate) fn normalize_incoming_messages(
    messages: Vec<NormalizedMessage>,
) -> Vec<NormalizedMessage> {
    filter_non_api_messages(merge_adjacent_assistant_messages(
        merge_adjacent_user_messages(normalize_system_local_command_messages(messages)),
    ))
    .into_iter()
    .map(normalize_message)
    .collect()
}

fn normalize_system_local_command_messages(
    messages: Vec<NormalizedMessage>,
) -> Vec<NormalizedMessage> {
    let mut result: Vec<NormalizedMessage> = Vec::new();

    for message in messages {
        let is_local_command = matches!(message.role, NormalizedRole::SystemEvent)
            && message.subtype.as_deref() == Some("local_command");

        if is_local_command {
            let user_message = NormalizedMessage {
                role: NormalizedRole::User,
                blocks: message.blocks,
                subtype: None,
            };

            match result.last_mut() {
                Some(last) if matches!(last.role, NormalizedRole::User) => {
                    last.blocks.extend(user_message.blocks);
                }
                _ => result.push(user_message),
            }
        } else {
            result.push(message);
        }
    }

    result
}

fn filter_non_api_messages(messages: Vec<NormalizedMessage>) -> Vec<NormalizedMessage> {
    messages
        .into_iter()
        .filter(|message| {
            !matches!(
                message.role,
                NormalizedRole::Progress | NormalizedRole::GroupedToolUse
            )
        })
        .collect()
}

fn merge_adjacent_user_messages(messages: Vec<NormalizedMessage>) -> Vec<NormalizedMessage> {
    let mut result: Vec<NormalizedMessage> = Vec::new();

    for message in messages {
        match (&message.role, result.last_mut()) {
            (NormalizedRole::User, Some(last)) if matches!(last.role, NormalizedRole::User) => {
                last.blocks.extend(message.blocks);
            }
            _ => result.push(message),
        }
    }

    result
}

fn merge_adjacent_assistant_messages(messages: Vec<NormalizedMessage>) -> Vec<NormalizedMessage> {
    let mut result: Vec<NormalizedMessage> = Vec::new();

    for message in messages {
        match (&message.role, result.last_mut()) {
            (NormalizedRole::Assistant, Some(last))
                if matches!(last.role, NormalizedRole::Assistant) =>
            {
                last.blocks.extend(message.blocks);
            }
            _ => result.push(message),
        }
    }

    result
}

fn normalize_message(message: NormalizedMessage) -> NormalizedMessage {
    match message.role {
        NormalizedRole::User => NormalizedMessage {
            role: message.role,
            blocks: normalize_error_tool_results(hoist_tool_results(message.blocks)),
            subtype: message.subtype,
        },
        _ => message,
    }
}

fn hoist_tool_results(blocks: Vec<NormalizedBlock>) -> Vec<NormalizedBlock> {
    let mut tool_results = Vec::new();
    let mut others = Vec::new();

    for block in blocks {
        match block {
            NormalizedBlock::ToolResult { .. } => tool_results.push(block),
            _ => others.push(block),
        }
    }

    tool_results.extend(others);
    tool_results
}

fn normalize_error_tool_results(blocks: Vec<NormalizedBlock>) -> Vec<NormalizedBlock> {
    blocks
        .into_iter()
        .map(|block| match block {
            NormalizedBlock::ToolResult {
                tool_use_id,
                content,
                is_error: true,
            } => NormalizedBlock::ToolResult {
                tool_use_id,
                content: json!(tool_result_content_as_text(&content)),
                is_error: true,
            },
            other => other,
        })
        .collect()
}

fn tool_result_content_as_text(content: &serde_json::Value) -> String {
    match content {
        serde_json::Value::String(text) => text.clone(),
        serde_json::Value::Array(items) => items
            .iter()
            .filter_map(|item| item.get("text").and_then(serde_json::Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => content.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn filters_progress_and_grouped_tool_use_messages() {
        let messages = vec![
            NormalizedMessage {
                role: NormalizedRole::Progress,
                blocks: vec![NormalizedBlock::Text("progress".to_string())],
                subtype: Some("task_progress".to_string()),
            },
            NormalizedMessage {
                role: NormalizedRole::GroupedToolUse,
                blocks: vec![NormalizedBlock::Text("grouped".to_string())],
                subtype: None,
            },
            NormalizedMessage {
                role: NormalizedRole::User,
                blocks: vec![NormalizedBlock::Text("hello".to_string())],
                subtype: None,
            },
        ];

        let normalized = normalize_incoming_messages(messages);
        assert_eq!(normalized.len(), 1);
        assert!(matches!(normalized[0].role, NormalizedRole::User));
    }

    #[test]
    fn merges_local_command_and_adjacent_user_messages_into_single_wire_user() {
        let messages = vec![
            NormalizedMessage {
                role: NormalizedRole::User,
                blocks: vec![NormalizedBlock::Text("hello".to_string())],
                subtype: None,
            },
            NormalizedMessage {
                role: NormalizedRole::SystemEvent,
                blocks: vec![NormalizedBlock::Text("local command".to_string())],
                subtype: Some("local_command".to_string()),
            },
            NormalizedMessage {
                role: NormalizedRole::User,
                blocks: vec![NormalizedBlock::ToolResult {
                    tool_use_id: "call_1".to_string(),
                    content: json!("result"),
                    is_error: false,
                }],
                subtype: None,
            },
        ];

        let normalized = normalize_incoming_messages(messages);
        assert_eq!(normalized.len(), 1);
        assert!(matches!(normalized[0].role, NormalizedRole::User));
    }

    #[test]
    fn converts_system_local_command_into_user_message() {
        let messages = vec![NormalizedMessage {
            role: NormalizedRole::SystemEvent,
            blocks: vec![NormalizedBlock::Text("stdout: hello".to_string())],
            subtype: Some("local_command".to_string()),
        }];

        let normalized = normalize_incoming_messages(messages);
        assert_eq!(normalized.len(), 1);
        assert!(matches!(normalized[0].role, NormalizedRole::User));
    }

    #[test]
    fn merges_system_local_command_into_adjacent_user() {
        let messages = vec![
            NormalizedMessage {
                role: NormalizedRole::User,
                blocks: vec![NormalizedBlock::Text("first".to_string())],
                subtype: None,
            },
            NormalizedMessage {
                role: NormalizedRole::SystemEvent,
                blocks: vec![NormalizedBlock::Text("stdout: hello".to_string())],
                subtype: Some("local_command".to_string()),
            },
        ];

        let normalized = normalize_incoming_messages(messages);
        assert_eq!(normalized.len(), 1);
        assert!(matches!(normalized[0].role, NormalizedRole::User));
        assert_eq!(normalized[0].blocks.len(), 2);
    }
}
