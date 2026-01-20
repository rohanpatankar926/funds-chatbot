import pandas as pd
import os
from typing import List, Dict, Optional
import json

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

class FundDataAnalyzer:
    
    def __init__(self, holdings_path: str = "holdings.csv", trades_path: str = "trades.csv"):
        self.holdings_path = holdings_path
        self.trades_path = trades_path
        self.holdings_df = None
        self.trades_df = None
        self.load_data()
    
    def load_data(self):
        try:
            self.holdings_df = pd.read_csv(self.holdings_path)
            self.trades_df = pd.read_csv(self.trades_path)
            
            if 'AsOfDate' in self.holdings_df.columns:
                self.holdings_df['AsOfDate'] = pd.to_datetime(
                    self.holdings_df['AsOfDate'], format='%d/%m/%y', errors='coerce'
                )
            if 'TradeDate' in self.trades_df.columns:
                self.trades_df['TradeDate'] = pd.to_datetime(
                    self.trades_df['TradeDate'], format='%H:%M.%S', errors='coerce'
                )
            
            print(f"Loaded {len(self.holdings_df)} holdings records")
            print(f"Loaded {len(self.trades_df)} trades records")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def get_holdings_count_by_fund(self, fund_name: Optional[str] = None) -> Dict:
        if fund_name:
            filtered = self.holdings_df[self.holdings_df['PortfolioName'] == fund_name]
            return {fund_name: len(filtered)}
        else:
            counts = self.holdings_df['PortfolioName'].value_counts().to_dict()
            return counts
    
    def get_trades_count_by_fund(self, fund_name: Optional[str] = None) -> Dict:
        if fund_name:
            filtered = self.trades_df[self.trades_df['PortfolioName'] == fund_name]
            return {fund_name: len(filtered)}
        else:
            counts = self.trades_df['PortfolioName'].value_counts().to_dict()
            return counts
    
    def get_fund_performance(self) -> pd.DataFrame:
        performance = self.holdings_df.groupby('PortfolioName').agg({
            'PL_YTD': 'sum',
            'MV_Base': 'sum',
            'Qty': 'sum'
        }).reset_index()
        
        performance = performance.sort_values('PL_YTD', ascending=False)
        performance.columns = ['Fund', 'Total_PL_YTD', 'Total_Market_Value', 'Total_Quantity']
        
        return performance
    
    def get_fund_summary(self, fund_name: str) -> Dict:
        holdings = self.holdings_df[self.holdings_df['PortfolioName'] == fund_name]
        trades = self.trades_df[self.trades_df['PortfolioName'] == fund_name]
        
        summary = {
            'Fund Name': fund_name,
            'Total Holdings': len(holdings),
            'Total Trades': len(trades),
            'Total PL_YTD': holdings['PL_YTD'].sum() if len(holdings) > 0 else 0,
            'Total Market Value (Base)': holdings['MV_Base'].sum() if len(holdings) > 0 else 0,
            'Unique Securities': holdings['SecurityId'].nunique() if len(holdings) > 0 else 0,
            'Custodians': holdings['CustodianName'].unique().tolist() if len(holdings) > 0 else [],
            'AsOfDate': holdings['AsOfDate'].max() if len(holdings) > 0 else None
        }
        
        return summary
    
    def get_all_funds(self) -> List[str]:
        holdings_funds = set(self.holdings_df['PortfolioName'].unique())
        trades_funds = set(self.trades_df['PortfolioName'].unique())
        return sorted(list(holdings_funds.union(trades_funds)))
    
    def search_holdings(self, query: str) -> pd.DataFrame:
        query_lower = query.lower()
        mask = (
            self.holdings_df['PortfolioName'].str.lower().str.contains(query_lower, na=False) |
            self.holdings_df['SecName'].str.lower().str.contains(query_lower, na=False) |
            self.holdings_df['SecurityTypeName'].str.lower().str.contains(query_lower, na=False)
        )
        return self.holdings_df[mask]
    
    def search_trades(self, query: str) -> pd.DataFrame:
        query_lower = query.lower()
        mask = (
            self.trades_df['PortfolioName'].str.lower().str.contains(query_lower, na=False) |
            self.trades_df['Name'].str.lower().str.contains(query_lower, na=False) |
            self.trades_df['Ticker'].str.lower().str.contains(query_lower, na=False)
        )
        return self.trades_df[mask]
    
    def get_data_overview(self) -> Dict:
        all_funds = self.get_all_funds()
        holdings_counts = self.get_holdings_count_by_fund()
        trades_counts = self.get_trades_count_by_fund()
        performance = self.get_fund_performance()
        
        overview = {
            'total_funds': len(all_funds),
            'funds': all_funds,
            'total_holdings': sum(holdings_counts.values()),
            'total_trades': sum(trades_counts.values()),
            'holdings_by_fund': holdings_counts,
            'trades_by_fund': trades_counts,
            'performance_summary': performance.to_dict('records') if len(performance) > 0 else []
        }
        
        if len(performance) > 0:
            overview['top_performer'] = performance.iloc[0].to_dict()
            overview['worst_performer'] = performance.iloc[-1].to_dict()
        
        return overview
    
    def get_top_securities(self, fund_name: Optional[str] = None, top_n: int = 10) -> pd.DataFrame:
        df = self.holdings_df
        if fund_name:
            df = df[df['PortfolioName'] == fund_name]
        
        top_securities = df.nlargest(top_n, 'MV_Base')[['SecName', 'SecurityId', 'MV_Base', 'PL_YTD', 'Qty', 'PortfolioName']]
        return top_securities
    
    def get_custodian_summary(self) -> Dict:
        custodian_summary = self.holdings_df.groupby('CustodianName').agg({
            'MV_Base': 'sum',
            'PL_YTD': 'sum',
            'PortfolioName': 'nunique'
        }).reset_index()
        custodian_summary.columns = ['Custodian', 'Total_Market_Value', 'Total_PL_YTD', 'Number_of_Funds']
        return custodian_summary.to_dict('records')
    
    def get_security_type_summary(self) -> Dict:
        if 'SecurityTypeName' not in self.holdings_df.columns:
            return {}
        type_summary = self.holdings_df.groupby('SecurityTypeName').agg({
            'MV_Base': 'sum',
            'PL_YTD': 'sum',
            'SecurityId': 'nunique'
        }).reset_index()
        type_summary.columns = ['Security_Type', 'Total_Market_Value', 'Total_PL_YTD', 'Unique_Securities']
        return type_summary.to_dict('records')


class FundChatbot:
    
    def __init__(self, analyzer: FundDataAnalyzer, api_key: Optional[str] = None):
        self.analyzer = analyzer
        self.api_key = api_key
        self.llm = None
        
        try:
            os.environ["GROQ_API_KEY"] = self.api_key
            self.llm = ChatGroq(model="qwen/qwen3-32b",temperature=0)
            print(f"Initialized chatbot with {self.llm.model_name}")
        except Exception as e:
            print(f"Warning: Could not initialize LLM: {e}")
            self.llm = None
    
    def _extract_fund_name(self, question: str) -> Optional[str]:
        funds = self.analyzer.get_all_funds()
        question_lower = question.lower()
        
        for fund in funds:
            if fund.lower() in question_lower:
                return fund
        
        return None
    
    def _generate_context(self, question: str) -> str:
        """Generate comprehensive context based on the question"""
        context_parts = []
        question_lower = question.lower()
        
        overview = self.analyzer.get_data_overview()
        context_parts.append("=== DATA OVERVIEW ===")
        context_parts.append(f"Total Funds: {overview['total_funds']}")
        context_parts.append(f"Total Holdings: {overview['total_holdings']}")
        context_parts.append(f"Total Trades: {overview['total_trades']}")
        context_parts.append(f"Available Funds: {', '.join(overview['funds'])}")
        
        fund_name = self._extract_fund_name(question)
        
        keywords = {
            'holdings': ['holding', 'holdings', 'position', 'positions', 'security', 'securities', 'asset', 'assets'],
            'trades': ['trade', 'trades', 'transaction', 'transactions', 'execution', 'executions'],
            'performance': ['performance', 'profit', 'loss', 'p&l', 'pl', 'ytd', 'return', 'returns', 'gain', 'losses'],
            'custodian': ['custodian', 'custodians', 'bank', 'banks'],
            'security_type': ['type', 'types', 'category', 'categories', 'class', 'classes'],
            'top': ['top', 'best', 'largest', 'highest', 'biggest'],
            'summary': ['summary', 'overview', 'details', 'information', 'info']
        }
        
        include_holdings = any(kw in question_lower for kw in keywords['holdings'])
        include_trades = any(kw in question_lower for kw in keywords['trades'])
        include_performance = any(kw in question_lower for kw in keywords['performance'])
        include_custodian = any(kw in question_lower for kw in keywords['custodian'])
        include_security_type = any(kw in question_lower for kw in keywords['security_type'])
        include_top = any(kw in question_lower for kw in keywords['top'])
        include_summary = any(kw in question_lower for kw in keywords['summary'])
        
        if not any([include_holdings, include_trades, include_performance, include_custodian, include_security_type]):
            include_holdings = True
            include_trades = True
            include_performance = True
            include_summary = True
        
        if include_holdings:
            context_parts.append("\n=== HOLDINGS INFORMATION ===")
            if fund_name:
                count = self.analyzer.get_holdings_count_by_fund(fund_name)
                context_parts.append(f"Total holdings for {fund_name}: {count.get(fund_name, 0)}")
                if include_top:
                    top_securities = self.analyzer.get_top_securities(fund_name, top_n=10)
                    context_parts.append(f"\nTop 10 securities by market value for {fund_name}:")
                    context_parts.append(top_securities.to_string(index=False))
            else:
                counts = self.analyzer.get_holdings_count_by_fund()
                context_parts.append(f"Holdings by fund: {json.dumps(counts, indent=2)}")
                if include_top:
                    top_securities = self.analyzer.get_top_securities(top_n=10)
                    context_parts.append(f"\nTop 10 securities by market value across all funds:")
                    context_parts.append(top_securities.to_string(index=False))
        
        if include_trades:
            context_parts.append("\n=== TRADES INFORMATION ===")
            if fund_name:
                count = self.analyzer.get_trades_count_by_fund(fund_name)
                context_parts.append(f"Total trades for {fund_name}: {count.get(fund_name, 0)}")
            else:
                counts = self.analyzer.get_trades_count_by_fund()
                context_parts.append(f"Trades by fund: {json.dumps(counts, indent=2)}")
        
        if include_performance:
            context_parts.append("\n=== PERFORMANCE INFORMATION ===")
            performance = self.analyzer.get_fund_performance()
            context_parts.append("Fund Performance (sorted by PL_YTD, highest first):")
            context_parts.append(performance.to_string(index=False))
            
            if len(performance) > 0:
                top_performer = performance.iloc[0]
                worst_performer = performance.iloc[-1]
                context_parts.append(f"\nTop Performer: {top_performer['Fund']} with PL_YTD: {top_performer['Total_PL_YTD']}")
                context_parts.append(f"Worst Performer: {worst_performer['Fund']} with PL_YTD: {worst_performer['Total_PL_YTD']}")
        
        if include_custodian:
            context_parts.append("\n=== CUSTODIAN INFORMATION ===")
            custodian_summary = self.analyzer.get_custodian_summary()
            context_parts.append(json.dumps(custodian_summary, indent=2, default=str))
        
        if include_security_type:
            context_parts.append("\n=== SECURITY TYPE INFORMATION ===")
            type_summary = self.analyzer.get_security_type_summary()
            if type_summary:
                context_parts.append(json.dumps(type_summary, indent=2, default=str))
        
        if fund_name or include_summary:
            context_parts.append("\n=== FUND DETAILS ===")
            if fund_name:
                summary = self.analyzer.get_fund_summary(fund_name)
                context_parts.append(f"Summary for {fund_name}:")
                context_parts.append(json.dumps(summary, indent=2, default=str))
            else:
                if include_summary and not fund_name:
                    for fund in overview['funds'][:5]:  # Limit to first 5 to avoid too much context
                        summary = self.analyzer.get_fund_summary(fund)
                        context_parts.append(f"\nSummary for {fund}:")
                        context_parts.append(json.dumps(summary, indent=2, default=str))
        
        return "\n".join(context_parts)
    
    def answer(self, question: str) -> str:
        context = self._generate_context(question)
        try:
            system_prompt = """You are a helpful and knowledgeable assistant that answers questions about fund holdings and trades data.

Your capabilities include:
- Answering questions about fund performance, holdings, trades, and portfolio composition
- Providing insights about specific funds or comparing multiple funds
- Analyzing security types, custodians, and market values
- Identifying top performers, largest positions, and trends

Guidelines:
- Use the provided context to answer questions accurately and comprehensively
- Be conversational and helpful, not just listing data
- When asked about performance, focus on PL_YTD (Year-to-Date Profit and Loss) values
- Always provide specific numbers when available
- If asked about a specific fund, provide detailed information about that fund
- If asked general questions, provide an overview across all funds
- Use natural language and explain what the numbers mean
- If the question is unclear, respond with 'Sorry can not find the answer'"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}\n\nPlease provide a helpful and detailed answer:")
            ]
            
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            # Fallback to rule-based if LLM fails
            raise e

def main():
    analyzer = FundDataAnalyzer()
    print(analyzer.get_all_funds())
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("\nNote: GROQ_API_KEY not set. Using rule-based responses.")
        api_key = None
    chatbot = FundChatbot(analyzer, api_key=api_key)
    while True:
        try:
            question = input("You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            print("\nBot: ", end="")
            answer = chatbot.answer(question)
            print(answer)
            print("\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


# if __name__ == "__main__":
#     main()


