def calculate_average(numbers):
    # Bug: crashes when the list is empty
    return sum(numbers) / len(numbers)


def get_first_item(items):
    # Bug: crashes when the list is empty
    return items[0]


def calculate_discount(price, discount_percent):
    # Missing validation: discounts above 100 produce a negative price
    return price - (price * discount_percent / 100)


def find_user(users, username):
    # Bug: returns "not found" before checking all users
    for user in users:
        if user["username"] == username:
            return user
        return "User not found"
