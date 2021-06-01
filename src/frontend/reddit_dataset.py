import time

import pandas as pd
import plotly.graph_objs as go
import streamlit as st

import SessionState

save_path = "C:/Users/TS/PycharmProjects/DS1-RecommendationSystems/data/reddit/"


@st.cache
def read_csv_cached(path):
    return pd.read_csv(path)


def app():
    @st.cache(show_spinner=False)
    def stats(df):
        return df.shape[0], df['user'].nunique(), df['subreddit'].nunique()

    t1 = time.time()

    raw_user_summary = read_csv_cached(save_path + "user_summary.csv")
    raw_subreddit_summary = read_csv_cached(save_path + "subreddit_summary.csv")
    subreddit_info = read_csv_cached(save_path + "subreddit_info.csv")

    t2 = time.time()
    print("{0:<20}{1:.3f}s".format("DATASET:", t2 - t1))

    state = SessionState.get(u_comments=20, u_reddit=20, r_comments=100, r_users=100, include_over18=False)

    @st.cache(show_spinner=False)
    def filter_dataset(u_comments, u_reddit, r_comments, r_users, include_over18):
        """

        :param u_comments: The minimum total number of comments a user should have.
        :param u_reddit: The minimum number of unique subreddits a user must have commented on.
        :param r_comments: The minimum number of comments a subreddit must have.
        :param r_users: The number of unique users a subreddit must have.
        :param include_over18:
        :return:
        """
        # filter user
        raw_dataset = read_csv_cached(save_path + "dataset.csv")
        progress.progress(20)

        this_users = raw_user_summary[(raw_user_summary.unique_subreddits >= u_reddit)
                                      & (raw_user_summary.total_num_comments >= u_comments)]
        progress.progress(40)
        this_subreddit = raw_subreddit_summary[
            (raw_subreddit_summary.unique_users >= r_users)
            & (raw_subreddit_summary.total_num_comments >= r_comments)]
        progress.progress(60)
        q = raw_dataset[(raw_dataset.user.isin(this_users.user)) &
                        (raw_dataset.subreddit.isin(this_subreddit.subreddit))]
        progress.progress(80)
        # remove all subreddits over 18
        if not include_over18:
            subreddits_under18 = q.subreddit.isin(subreddit_info[~subreddit_info.over18].subreddit)
            q = q[subreddits_under18]
        progress.progress(100)
        return q

    @st.cache
    def group_subreddit(df, limit=15, sort="by_user"):
        subreddit_group = df.groupby(by=['subreddit'])['count']
        sum_comments_per_subreddit = subreddit_group.sum().reset_index(name="total_num_comments")
        unique_users_per_subreddit = subreddit_group.count().reset_index(name="unique_users")
        subreddit_summary = sum_comments_per_subreddit.merge(unique_users_per_subreddit, on="subreddit")
        sort_by_col = "unique_users" if sort == "by user" else "total_num_comments"
        subreddit_summary = subreddit_summary.sort_values(by=sort_by_col, ascending=False, ignore_index=True)
        subreddit_summary = subreddit_summary.iloc[:limit]

        layout = go.Layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           height=400, width=400, margin=dict(l=10, r=10, t=10, b=10),
                           yaxis=dict(title="Total number of comments"),
                           yaxis2={'title': 'Number unique users', 'overlaying': 'y', 'side': 'right'})
        fig = go.Figure(layout=layout)
        fig.add_trace(go.Bar(x=subreddit_summary['subreddit'], opacity=1, name="Number of comments",
                             y=subreddit_summary['total_num_comments'], orientation='v', offsetgroup=1,
                             yaxis='y'))
        fig.add_trace(go.Bar(x=subreddit_summary['subreddit'], opacity=1, name="Unique users",
                             y=subreddit_summary['unique_users'], orientation='v', offsetgroup=2, yaxis='y2'))
        return fig

    @st.cache
    def group_user(df, limit=15, sort="total_num_comments"):
        subreddit_group = df.groupby(by=['user'])['count']
        sum_comments_per_subreddit = subreddit_group.sum().reset_index(name="total_num_comments")
        unique_users_per_subreddit = subreddit_group.count().reset_index(name="unique_subreddits")
        user_summary = sum_comments_per_subreddit.merge(unique_users_per_subreddit, on="user")
        sort_by_col = "unique_subreddits" if sort == "by subreddits" else "total_num_comments"
        user_summary = user_summary.sort_values(by=sort_by_col, ascending=False, ignore_index=True)
        user_summary = user_summary.iloc[:limit]

        layout = go.Layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           height=400, width=400, margin=dict(l=10, r=10, t=10, b=10),
                           yaxis=dict(title="Total number of comments"),
                           yaxis2={'title': 'Number unique users', 'overlaying': 'y', 'side': 'right'})
        fig = go.Figure(layout=layout)
        fig.add_trace(go.Bar(x=user_summary['user'], opacity=1, name="Number of comments",
                             y=user_summary['total_num_comments'], orientation='v', offsetgroup=1,
                             yaxis='y'))
        fig.add_trace(go.Bar(x=user_summary['user'], opacity=1, name="Unique users",
                             y=user_summary['unique_subreddits'], orientation='v', offsetgroup=2, yaxis='y2'))
        return fig

    @st.cache
    def subreddit_details(subreddit):
        return subreddit_info[subreddit_info['subreddit'] == subreddit].to_numpy()[0]

    t3 = time.time()
    # Sidebar
    title_container = st.sidebar.empty()
    s_r_users = st.sidebar.slider("Min. User per Subreddit",
                                  value=state.r_users, min_value=0, max_value=200,
                                  help="The minimum number of unique users a subreddit must have to be included in the "
                                       "dataset.")
    s_r_comments = st.sidebar.slider("Min. Comment per Subreddit",
                                     value=state.r_comments, min_value=0, max_value=200,
                                     help="The minimum number of comments a subreddit must have to be included in the "
                                          "dataset.")
    s_u_comments = st.sidebar.slider("Min. Comment per User",
                                     value=state.u_comments, min_value=0, max_value=200,
                                     help="The minimum number total number of comments a user must have written,")
    s_u_reddit = st.sidebar.slider("Min. Subreddit per User",
                                   value=state.u_reddit, min_value=0, max_value=200,
                                   help="The number of different subreddits that a user must have commented on to be "
                                        "included in the dataset.")

    s_include_over18 = st.sidebar.checkbox("Include subreddits over 18?", value=state.include_over18,
                                           help="Should subreddits be included that are not approved for minors?")
    progress = st.sidebar.progress(0)

    _, _, sidebar_but2 = st.sidebar.beta_columns([1, 2, 1])

    if sidebar_but2.button("Apply"):
        state.r_users = s_r_users
        state.r_comments = s_r_comments
        state.u_comments = s_u_comments
        state.u_reddit = s_u_reddit
        state.include_over18 = s_include_over18

    same = (state.r_users == s_r_users) & (state.r_comments == s_r_comments) & (
            state.u_comments == s_u_comments) & (state.u_reddit == s_u_reddit) & (
                   state.include_over18 == s_include_over18)

    title_container.title("Configuration {0}".format("🟢" if same else "🔴"))
    t4 = time.time()
    print("{0:<20}{1:.3f}s".format("SIDEBAR:", t4 - t3))

    data = filter_dataset(r_users=state.r_users, r_comments=state.r_comments,
                          u_comments=state.u_comments, u_reddit=state.u_reddit, include_over18=state.include_over18)
    t5 = time.time()
    print("{0:<20}{1:.3f}s".format("FILTER:", t5 - t4))

    st.title("Subreddit Recommender")
    col1, col2 = st.beta_columns([1, 1])
    col1.text("Information about Reddit ...")
    with col2:
        data_size, num_users, num_reddits = stats(data)
        st.text("{0:<20}{1:,}".format("Total datapoints:", data_size))
        st.text("{0:<20}{1:,}".format("Unique users:", num_users))
        st.text("{0:<20}{1:,}".format("Unique reddits:", num_reddits))

    st.subheader("Histogram of Ratings")
    hist_col1, hist_col2 = st.beta_columns([1, 2])

    hist_col1.write(data['count'].describe())

    @st.cache
    def histogram():
        print("HISTOGRAM")
        layout = go.Layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           height=250, width=400, margin=dict(l=10, r=10, t=10, b=10))
        return go.Figure(data=go.Histogram(x=data['count'], nbinsx=50), layout=layout)

    t5 = time.time()
    hist_col2.plotly_chart(histogram(), use_container_width=True)
    t6 = time.time()
    print("{0:<20}{1:.3f}s".format("HISTOGRAM:", t6 - t5))

    st.write('<style>.st-db{flex-direction:row;}</style>', unsafe_allow_html=True)

    c1, c2, c3 = st.beta_columns([2, 1, 1])
    c1.subheader("Top Subreddits")
    n_top_subreddits = c2.number_input("Number of subreddits", value=15, min_value=2, max_value=100, step=1)
    sort_column_subreddit = c3.radio("Sort", options=["by comments", "by user"], key="sort_reddit",
                                     help="Should the top users be displayed sorted by the number of comments or the "
                                          "number of unique subreddits they commented on.")
    st.plotly_chart(group_subreddit(data, limit=n_top_subreddits, sort=sort_column_subreddit), use_container_width=True)

    c1, c2, c3 = st.beta_columns([2, 1, 1])
    c1.subheader("Top User")
    n_top_users = c2.number_input("Number of users", value=15, min_value=2, max_value=100, step=1)
    sort_column_user = c3.radio("Sort", options=["by comments", "by subreddits"],
                                help="Should the top users be displayed sorted by the number of comments or the number "
                                     "of unique subreddits they commented on.")
    st.plotly_chart(group_user(data, limit=n_top_users, sort=sort_column_user), use_container_width=True)
    t7 = time.time()
    print("{0:<20}{1:.3f}s".format("CHARTS:", t7 - t6))

    # st.subheader("Subreddit and User details")
    # with st.beta_expander("Subreddit details"):
    #     detail_subreddit = st.selectbox("Subreddit", options=data['subreddit'].unique())
    #     _, num_subscribers, over18, public_description = subreddit_details(subreddit=detail_subreddit)
    #     st.subheader("Description")
    #     st.markdown(public_description)
    #     st.markdown("{0:,} Subscribers".format(num_subscribers))
    #     st.markdown("{0}".format("🔞" if over18 else "No age restriction."))
    #     st.markdown(f"[Link to forum](https://www.reddit.com/r/{detail_subreddit}/)")
    #
    # with st.beta_expander("User details"):
    #     detail_user = st.selectbox("User", options=data['user'].unique())
    #     st.write(data[data['user'] == detail_user][['subreddit', 'count']])


if __name__ == '__main__':
    start = time.time()
    app()
    print("-"*26)
    print("{0:<20}{1:.3f}s".format("TOTAL:", time.time() - start))
    print()
