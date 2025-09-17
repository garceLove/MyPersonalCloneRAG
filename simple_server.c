#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>

void handle_client(int client_sock);

int main(int argc, char *argv[]) {
    int server_sock, client_sock;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_addr_len = sizeof(client_addr);

    // 创建 socket
    server_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (server_sock == -1) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    // 设置地址和端口
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY; // 监听所有网络接口
    server_addr.sin_port = htons(8080);       // 端口号

    // 绑定 socket
    if (bind(server_sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) == -1) {
        perror("Bind failed");
        exit(EXIT_FAILURE);
    }

    // 监听连接
    if (listen(server_sock, 5) == -1) {
        perror("Listen failed");
        exit(EXIT_FAILURE);
    }
    printf("Server listening on port 8080...\n");

    // 接受客户端连接
    while (1) {
        client_sock = accept(server_sock, (struct sockaddr *)&client_addr, &client_addr_len);
        if (client_sock == -1) {
            perror("Accept failed");
            continue;
        }

        // 处理客户端请求
        handle_client(client_sock);
        close(client_sock); // 关闭连接
    }

    close(server_sock);
    return 0;
}

void handle_client(int client_sock) {
    char buffer[1024] = {0};
    // 简单读取 HTTP 请求头，但此处不解析
    read(client_sock, buffer, 1023);
    // printf("Received request:\n%s\n", buffer);

    // 构建 HTTP 响应
    const char *response_body = "<h1>Hello, World!</h1>";
    const char *http_response_template =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/html\r\n"
        "Content-Length: %d\r\n"
        "Connection: close\r\n"
        "\r\n"
        "%s";

    char http_response[2048];
    int content_length = strlen(response_body);
    sprintf(http_response, http_response_template, content_length, response_body);

    // 发送响应
    write(client_sock, http_response, strlen(http_response));
}
