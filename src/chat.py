from search import search_prompt


def main():
    print("Chat iniciado. Digite 'sair' para encerrar.\n")

    while True:
        pergunta = input("PERGUNTA: ").strip()

        if pergunta.lower() in ("sair", "exit", "quit"):
            break

        if not pergunta:
            continue

        result = search_prompt(question=pergunta)

        if not result:
            print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
            return

        chain, contexto = result
        resposta = chain.invoke({"contexto": contexto, "pergunta": pergunta})
        print(f"RESPOSTA: {resposta.content}\n")


if __name__ == "__main__":
    main()
