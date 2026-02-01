#Importing the necessary packages from the chromadb
#embedding_functions is used for defining embeddings

import chromadb
from chromadb.utils import embedding_functions

#define the mebedding function from Sentence Tranformers
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

#create an instance of chromadb client
client = chromadb.Client()

#defining name for the collection where data will be stored and accessed
#it is also used to group records
collection_name = "books_collection"

#creating the main function where the main logic is performed
#including creating collections, embeddings and similarity search
def main():
    try:
        # defining a list of books dictionaries
        # list of book dictionaries with comprehensive details for advanced search
        books = [
            {
                "id": "book_1",
                "title": "The Great Gatsby",
                "author": "F. Scott Fitzgerald",
                "genre": "Classic",
                "year": 1925,
                "rating": 4.1,
                "pages": 180,
                "description": "A tragic tale of wealth, love, and the American Dream in the Jazz Age",
                "themes": "wealth, corruption, American Dream, social class",
                "setting": "New York, 1920s"
            },
            {
                "id": "book_2",
                "title": "To Kill a Mockingbird",
                "author": "Harper Lee",
                "genre": "Classic",
                "year": 1960,
                "rating": 4.3,
                "pages": 376,
                "description": "A powerful story of racial injustice and moral growth in the American South",
                "themes": "racism, justice, moral courage, childhood innocence",
                "setting": "Alabama, 1930s"
            },
            {
                "id": "book_3",
                "title": "1984",
                "author": "George Orwell",
                "genre": "Dystopian",
                "year": 1949,
                "rating": 4.4,
                "pages": 328,
                "description": "A chilling vision of totalitarian control and surveillance society",
                "themes": "totalitarianism, surveillance, freedom, truth",
                "setting": "Oceania, dystopian future"
            },
            {
                "id": "book_4",
                "title": "Harry Potter and the Philosopher's Stone",
                "author": "J.K. Rowling",
                "genre": "Fantasy",
                "year": 1997,
                "rating": 4.5,
                "pages": 223,
                "description": "A young wizard discovers his magical heritage and begins his education at Hogwarts",
                "themes": "friendship, courage, good vs evil, coming of age",
                "setting": "England, magical world"
            },
            {
                "id": "book_5",
                "title": "The Lord of the Rings",
                "author": "J.R.R. Tolkien",
                "genre": "Fantasy",
                "year": 1954,
                "rating": 4.5,
                "pages": 1216,
                "description": "An epic fantasy quest to destroy a powerful ring and save Middle-earth",
                "themes": "heroism, friendship, good vs evil, power corruption",
                "setting": "Middle-earth, fantasy realm"
            },
            {
                "id": "book_6",
                "title": "The Hitchhiker's Guide to the Galaxy",
                "author": "Douglas Adams",
                "genre": "Science Fiction",
                "year": 1979,
                "rating": 4.2,
                "pages": 224,
                "description": "A humorous space adventure following Arthur Dent across the galaxy",
                "themes": "absurdity, technology, existence, humor",
                "setting": "Space, various planets"
            },
            {
                "id": "book_7",
                "title": "Dune",
                "author": "Frank Herbert",
                "genre": "Science Fiction",
                "year": 1965,
                "rating": 4.3,
                "pages": 688,
                "description": "A complex tale of politics, religion, and ecology on a desert planet",
                "themes": "power, ecology, religion, politics",
                "setting": "Arrakis, distant future"
            },
            {
                "id": "book_8",
                "title": "The Hunger Games",
                "author": "Suzanne Collins",
                "genre": "Dystopian",
                "year": 2008,
                "rating": 4.2,
                "pages": 374,
                "description": "A teenage girl fights for survival in a brutal televised competition",
                "themes": "survival, oppression, sacrifice, rebellion",
                "setting": "Panem, dystopian future"
            },
        ]

        #creating a collection to store books data
        collection = client.create_collection(
            name = collection_name, #name of the collection
            metadata = {"description": "a collection to store various books"},
            configuration={
                "hnsw":{"space":"cosine"},
                "embedding_function":ef
            }
        )

        print(f"Collection created: {collection.name}")

        #Creating a compehensive document consisting of descriptions of books in plain English.
        library = create_documents(books)

        #Adding the library to the collection in the chromadb database
        collection.add(
            #adding ids as identifiers
            ids = [book["id"] for book in books ],

            #adding the library
            documents = library,

            #constructing some metadata about each book
            metadatas = [{
                "title":book["title"],
                "author":book["author"],
                "year":book["year"],
                "rating":book["rating"],
                "genre":book["genre"]
            } for book in books]

        )

        print("Done adding collections and metadatas...")

        #Retrieving all documents from the collection to verify
        all_books = collection.get()

        print("Collection contents:")
        print(f"Number of books = {len(all_books['documents'])}")

        advanced_search(collection, all_books)
        
    except Exception as error:
        # Catching and handling any errors that occur within the 'try' block
        # Logs the error message to the console for debugging purposes
        print(f"Error: {error}")   


def create_documents(books):
    print("Creating documents out of books...")
    library=[]
    for book in books:
        doc = f"{book['title']} written by {book['author']} published in {book['year']}."
        doc += f"The book is of {book['genre']} genre and has a rating of {book['rating']}."
        doc += f"The book consists of {book['pages']} pages and is described as {book['description']}."
        doc += f"The story was set in {book['setting']} and has the following themes {book['themes']}."
        library.append(doc)

    print("Done...")
    return library

def advanced_search(collection, all_books):
    try:
        print("===Similarity Search for Book Recommendations===")

        print("\nSearching for MAGICAL FANTASY ADVENTURE.")
        query="Recommend a book with magical fantasy adventure theme."
        results = collection.query(
            query_texts=[query],
            n_results=3
        )
        for i, (id, book, distance) in enumerate(zip(
            results['ids'][0], results['documents'][0], results['distances'][0]
        )):
            metadata = results['metadatas'][0][i]
            print(f"{i+1}. {metadata['title']} ({id}) - distance:{distance:.4f}")
            print(f"Author: {metadata['author']}, Genre: {metadata['genre']}, Year: {metadata['year']}")
            #print(f"Book: {book[:100]}")


        query = "Books published in 1980s"
        results = collection.query(
            query_texts=[query],
            n_results=3
        )
        for i, (id, book, distance) in enumerate(zip(
            results['ids'][0], results['documents'][0], results['distances'][0]
        )):
            metadata = results['metadatas'][0][i]
            print(f"{i+1}. {metadata['title']} ({id}) - distance:{distance:.4f}")
            print(f"Author: {metadata['author']}, Genre: {metadata['genre']}, Year: {metadata['year']}")
            #print(f"Book: {book[:100]}")

        print("===Metadata filtering by genre (Fantasy or Science Fiction)===")

        results = collection.get(
            where = {"genre": {"$in": ["Fantasy", "Science Fiction"]}}
        )
        print(f"Found {len(results['ids'])} books in Fantasy or Science Fiction")

        for i, id in enumerate(results['ids']):
            metadata = results['metadatas'][i]
            print(f" - {metadata['title']} : {metadata['genre']}")


        results = collection.get(
            where = {
                "$and" :[
                    {"year" : {"$gte" : 1980}},
                    {"year" : {"$lte" : 1990}}
                ]
            }
        )

        print(f"Found {len(results['ids'])} books published in 1980s")

        for i, id in enumerate(results['ids']):
            metadata = results['metadatas'][i]
            print(f" - {metadata['title']} : {metadata['year']}")

        print(" === Filtering books by rating (4.0 or higher) === ")

        results = collection.get(
            where = {"rating": {"$gte": 4.0}}
        )
        print(f"Found {len(results['ids'])} books in with rating 4.0 or higher")
        for i, id in enumerate(results['ids']):
            metadata = results['metadatas'][i]
            print(f" - {metadata['title']} : {metadata['rating']}")


        print(" === Combined Search: Similarity + Metadata Filtering === ")
        query = "books with dystopian theme or genre"

        results = collection.query(
            query_texts = [query],
            n_results = 3,
            where = {"rating": {"$gte" : 4.3}}
        )

        print(f"Found {len(results['ids'])} dystopian books with rating 4.0 or higher.")

        for i, (doc_id, document, distance) in enumerate(zip(
                    results['ids'][0], results['documents'][0], results['distances'][0]
                )):
                    metadata = results['metadatas'][0][i]
                    print(f"  {i+1}. {metadata['title']} ({doc_id}) - Distance: {distance:.4f}")
                    print(f"     {metadata['genre']} with rating {metadata['rating']}")
                    print(f"     Document snippet: {document[:80]}...")

        # Check if the results are empty or undefined
        if not results or not results['ids'] or len(results['ids'][0]) == 0:
            # Log a message if no similar documents are found for the query term
            print(f'No documents found similar to "{query}"')
            return

    except Exception as error:
        print(f"Error in advanced search: {error}")


if __name__ == "__main__":
    main()